"""Синтез русской речи по тексту с моделями из ЛР1-4"""

import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LAB_DIR = Path(__file__).resolve().parent
ROOT = LAB_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIL = "<SIL>"
SAMPLE_RATE = 22050
VOCODER_SOURCE = "speechbrain/tts-hifigan-libritts-22050Hz"
DEFAULT_RUN = LAB_DIR.parent / "lab4_acoustic/runs_epochonly_full/baseline_gt"
HIFIGAN_AUDIO = dict(sample_rate=SAMPLE_RATE, fft_size=1024, win_length=1024,
                     hop_length=256, num_mels=80, mel_fmin=0.0, mel_fmax=8000.0,
                     spec_gain=1.0, log_func="np.log", signal_norm=False,
                     do_amp_to_db_mel=True, do_trim_silence=False, preemphasis=0.0)
WORD = re.compile(r"[А-ЯЁа-яё][А-ЯЁа-яё\u0301]*(?:-[А-ЯЁа-яё][А-ЯЁа-яё\u0301]*)*")
PUNCTUATION = frozenset(".,?!:;…-–—«»„“”\"'()[]{}")
MARKUP_RE = re.compile(r'<pause\s+type\s*=\s*"(normal|none)"\s*/?>')


def strip_markup(text):
    requests = {}
    out = []
    cursor = 0
    for match in MARKUP_RE.finditer(text):
        out.append(text[cursor:match.start()])
        n_words = len(re.findall(r"\w+", "".join(out)))
        if n_words:
            requests[n_words - 1] = match.group(1)
        out.append(" ")
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out), requests


def load_vocoder(device, source=VOCODER_SOURCE):
    from speechbrain.inference.vocoders import HIFIGAN
    from speechbrain.utils.fetching import LocalStrategy

    savedir = LAB_DIR / "pretrained_models/hifigan"
    if source == VOCODER_SOURCE and all((savedir / n).is_file()
                                        for n in ("generator.ckpt", "hyperparams.yaml")):
        source = str(savedir)
    return HIFIGAN.from_hparams(source=source, savedir=str(savedir) if source == VOCODER_SOURCE else None,
                                local_strategy=LocalStrategy.COPY, run_opts={"device": device})


def lookup_key(word):
    letters = unicodedata.normalize("NFD", word.lower()).replace("\u0301", "")
    return unicodedata.normalize("NFC", letters)


def decode_mel(vocoder, mel, device, hop_length):
    import torch
    with torch.inference_mode():
        tensor = torch.from_numpy(mel).unsqueeze(0).to(device)
        signal = vocoder.decode_batch(tensor)[0, 0]
    # HiFi-GAN дополняет mel с двух сторон и возвращает эти кадры вместе со звуком
    pad = vocoder.mods.generator.inference_padding * hop_length
    length = mel.shape[-1] * hop_length
    if signal.numel() != length + 2 * pad:
        raise ValueError("Длина выхода HiFi-GAN расходится с шагом mel")
    return signal[pad:pad + length].cpu().numpy()


@dataclass
class Timings:
    stages: dict[str, float] = field(default_factory=dict)
    audio_seconds: float = 0.0
    chunks: int = 0

    @property
    def total(self):
        return sum(self.stages.values())

    @property
    def rtf(self):
        return self.total / self.audio_seconds if self.audio_seconds else float("inf")

    def report(self):
        lines = [f"{name:12s} {seconds * 1000:8.1f} мс" for name, seconds in self.stages.items()]
        lines.append(f"Всего {self.total:.3f} с, звук {self.audio_seconds:.2f} с, RTF {self.rtf:.3f}")
        return "\n".join(lines)


class TTSPipeline:
    def __init__(self, run_dir=DEFAULT_RUN, device="cuda", vocoder_source=VOCODER_SOURCE):
        import torch
        from labs.lab1_text.text_normalizer import TextNormalizer
        from labs.lab2_pauses.pause_predictor import PausePredictor, get_spacy
        from labs.lab3_g2p.g2p import G2P, load_corpus_lexicon, load_mfa_lexicon
        from TTS.config import load_config
        from TTS.tts.configs.shared_configs import CharactersConfig
        from TTS.tts.models.forward_tts import ForwardTTS
        from TTS.tts.utils.text.tokenizer import TTSTokenizer
        from TTS.utils.audio import AudioProcessor

        self.device = device
        configs = list(Path(run_dir).rglob("config.json"))
        if len(configs) != 1:
            raise ValueError(f"Нужна папка одного запуска FastSpeech2, найдено конфигураций: {len(configs)}")
        config_path = configs[0]
        numbered = list(config_path.parent.glob("best_model_[0-9]*.pth"))
        self.checkpoint = (max(numbered, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
                           if numbered else config_path.with_name("best_model.pth"))
        config = load_config(str(config_path))
        if vocoder_source == VOCODER_SOURCE:
            for key, expected in HIFIGAN_AUDIO.items():
                if getattr(config.audio, key) != expected:
                    raise ValueError(f"Для HiFi-GAN нужно {key}={expected}, в ЛР4 {getattr(config.audio, key)}")
        saved_chars = json.loads(config_path.read_text(encoding="utf-8"))["characters"]
        if saved_chars is None:
            # В этом запуске поле characters пустое. Повторяю настройку из evaluate.py ЛР4
            inventory = (ROOT / "common/phones_mfa.txt").read_text(encoding="utf-8").splitlines()
            saved_chars = dict(characters_class="TTS.tts.utils.text.characters.MulticharPhonemes",
                               characters=[unicodedata.normalize("NFC", p.strip()) for p in inventory if p.strip()],
                               punctuations="", pad="<PAD>", eos="<EOS>", bos="<BOS>", blank="<BLNK>",
                               is_unique=False, is_sorted=True)
        config.characters = CharactersConfig(**saved_chars)
        self.phone_set = set(config.characters.characters)
        self.ap = AudioProcessor.init_from_config(config)
        self.tokenizer, self.config = TTSTokenizer.init_from_config(config)
        self.model = ForwardTTS(self.config, self.ap, self.tokenizer, speaker_manager=None)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        del state
        self.model.eval().to(device)
        self.vocoder = load_vocoder(device, vocoder_source)
        self.normalizer = TextNormalizer()
        self.pauses = PausePredictor(LAB_DIR.parent / "lab2_pauses/models")
        if self.pauses.approach == "morph_syntax":
            get_spacy()
        data = LAB_DIR.parent / "lab3_g2p/data"
        corpus = load_corpus_lexicon(data / "lexicon.csv")
        mfa = load_mfa_lexicon(data / "russian_mfa.dict", self.phone_set)
        self.g2p = G2P(corpus, mfa, allowed_phones=self.phone_set, oov_mode="final")

    def normalize(self, text):
        return self.normalizer.normalize(text)

    def tokenize(self, text):
        tokens = []
        start = 0
        for match in WORD.finditer(text):
            gap = self._punctuation(text[start:match.start()])
            if tokens:
                tokens[-1] += gap
            tokens.append(match.group())
            start = match.end()
        tail = self._punctuation(text[start:])
        if tokens:
            tokens[-1] += tail
        return tokens

    @staticmethod
    def _punctuation(gap):
        compact = "".join(gap.split())
        unknown = set(compact) - PUNCTUATION
        if unknown:
            raise ValueError("Запишите словами неподдерживаемые символы: " + " ".join(sorted(unknown)))
        return compact

    def insert_pauses(self, tokens, requests=None):
        if not tokens:
            return []
        flags, _ = self.pauses.predict(tokens)
        result = []
        for i, token in enumerate(tokens):
            result.append(token)
            if i + 1 < len(tokens) and flags[i]:
                result.append(SIL)
        return result

    def to_phones(self, tokens):
        words = []
        for token in tokens:
            if token == SIL:
                words.append(SIL)
                continue
            word = lookup_key(WORD.match(token).group())
            if "-" in word and self.g2p.source(word) == "oov":
                words.extend(word.split("-"))
            else:
                words.append(word)
        return [p for pronunciation in self.g2p.transcribe(words) for p in pronunciation]

    @staticmethod
    def groups(tokens, limit=18):
        # Ограничиваю длину перед FastSpeech2. По возможности режу на паузе или знаке
        start = 0
        while start < len(tokens):
            end = min(start + limit, len(tokens))
            if end < len(tokens):
                for idx in range(end - 1, start, -1):
                    if tokens[idx] == SIL or tokens[idx].endswith((",", ".", "!", "?", ";", ":")):
                        end = idx + 1
                        break
                if end < len(tokens) and tokens[end] == SIL:
                    end += 1
            yield tokens[start:end]
            start = end

    def mel(self, phones):
        import torch
        unknown = set(phones) - self.phone_set
        if unknown:
            raise ValueError(f"Фонемы вне инвентаря: {sorted(unknown)}")
        with torch.inference_mode():
            out = self.model.synthesize(" ".join(phones), use_griffin_lim=False)
        return out["outputs"]["model_outputs"][0].detach().cpu().numpy().T

    def vocode(self, mel):
        return decode_mel(self.vocoder, mel, self.device, self.ap.hop_length)

    def __call__(self, text, markup=False):
        timings = Timings()

        def timed(name, function, *args):
            start = time.perf_counter()
            result = function(*args)
            timings.stages[name] = timings.stages.get(name, 0) + time.perf_counter() - start
            return result

        text = unicodedata.normalize("NFC", str(text or ""))
        if markup:
            raise ValueError("Конвейер принимает обычный текст без управляющих тегов")
        normalized = timed("normalize", self.normalize, text)
        tokens = timed("tokenize", self.tokenize, normalized)
        with_pauses = timed("pauses", self.insert_pauses, tokens)
        parts = []
        for group in self.groups(with_pauses):
            phones = timed("g2p", self.to_phones, group)
            if phones:
                mel = timed("acoustic", self.mel, phones)
                parts.append(timed("vocoder", self.vocode, mel))
        wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        timings.audio_seconds = len(wav) / SAMPLE_RATE
        timings.chunks = len(parts)
        return wav, timings

    def save(self, wav, path):
        self.ap.save_wav(wav, str(path), sr=SAMPLE_RATE)

