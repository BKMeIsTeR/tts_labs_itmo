"""Копи-синтез записи RUSLAN с параметрами звука из ЛР4"""

import argparse
from pathlib import Path

import numpy as np
from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor

from tts_pipeline import HIFIGAN_AUDIO, SAMPLE_RATE, decode_mel, load_vocoder


def check_vocoder(wav_path, output_dir, device="cuda"):
    config = BaseAudioConfig(**HIFIGAN_AUDIO, ref_level_db=20, trim_db=60.0)
    ap = AudioProcessor.init_from_config(config)
    wav = ap.load_wav(str(wav_path))
    mel = ap.melspectrogram(wav)
    vocoder = load_vocoder(device)
    restored = decode_mel(vocoder, mel, device, ap.hop_length)[:len(wav)]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # У Гриффина-Лима случайная начальная фаза
    np.random.seed(42)
    for name, signal in [("natural", wav), ("copysynth_hifigan", restored),
                         ("copysynth_griffinlim", ap.inv_melspectrogram(mel))]:
        ap.save_wav(signal, str(output_dir / f"{name}.wav"), sr=SAMPLE_RATE)
    return mel.shape, float(mel.min()), float(mel.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--out", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(check_vocoder(args.wav, args.out, args.device))
