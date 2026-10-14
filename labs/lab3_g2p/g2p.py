"""Grapheme-to-phoneme front end — skeleton for lab 3.

Run as a script to score the module on the prepared data::

    python g2p.py

Two phone error rates are reported on the split carried in the data's own `set` column: PER over
whole `test` utterances — the front end as lab 5 will run it — and PER over `test` tokens
absent from the training lexicon, which is the out-of-vocabulary model alone.
"""

from __future__ import annotations

import csv
import math
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


TOKENS_PHONES_DATA = Path("data/tokens_phones.csv")
CORPUS_LEXICON_PATH = Path("data/lexicon.csv")
MFA_LEXICON_PATH = Path("data/russian_mfa.dict")

PHONE_PATHS = [
    Path("common/phones_mfa.txt"),
    Path("configs/phones_mfa.txt"),
]

SIL = "<SIL>"
LENGTH_MARK = "ː"

VOWEL_LETTERS = set("аеёиоуыэюя")
VOWEL_PHONES = {
    "a", "æ", "e", "ɛ", "i", "o", "ɵ",
    "u", "ʉ", "ɐ", "ə", "ɪ", "ʊ", "ɨ",
}

# По этим вариантам гласных приблизительно определяю ударный слог
# простая эвристика по train
STRESSED_VOWELS = {
    "a", "æ", "e", "ɛ", "i", "o", "ɵ", "u", "ʉ",
}


def normalize_word(word: str) -> str:
    """Привести слово к одному Unicode-виду и нижнему регистру"""

    return unicodedata.normalize("NFC", str(word)).lower().strip()


def normalize_phone(phone: str) -> str:
    """Привести обозначение фонемы к виду из phones_mfa.txt"""

    phone = unicodedata.normalize("NFC", str(phone))
    return phone.replace(LENGTH_MARK, "")


def load_allowed_phones() -> set[str] | None:
    """Прочитать список фонем, которые разрешены по условию"""

    for path in PHONE_PATHS:
        if not path.exists():
            continue

        phones = set()
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                phones.update(normalize_phone(phone) for phone in line.split())

        phones.add(SIL)

        print(f"Список фонем: {path}")
        print(f"Количество фонем: {len(phones)}")
        return phones

    print("Внимание: phones_mfa.txt не найден")
    print("Проверка списка фонем будет пропущена")
    return None


def load_tokens(path: Path = TOKENS_PHONES_DATA) -> pd.DataFrame:
    """Прочитать подготовленную таблицу и убрать строки без нормальной транскрипции"""

    tokens = pd.read_csv(path, sep="|", quoting=csv.QUOTE_NONE)
    tokens["token"] = tokens["token"].astype(str).map(normalize_word)
    tokens["phones"] = tokens["phones"].fillna("").astype(str)

    bad = (
        (tokens["phones"] == "")
        | tokens["phones"].str.split().map(lambda phones: "spn" in phones)
    )

    if bad.any():
        print(f"Удаляем {bad.sum()} строк без нормальной транскрипции")

    return tokens[~bad].reset_index(drop=True)


def load_corpus_lexicon(
    path: Path = CORPUS_LEXICON_PATH,
) -> dict[str, Counter]:
    """Прочитать lexicon.csv, который был собран только по train"""

    lexicon: dict[str, Counter] = defaultdict(Counter)
    df = pd.read_csv(path, sep="|", quoting=csv.QUOTE_NONE)

    for row in df.itertuples(index=False):
        word = normalize_word(row.word)
        phones = " ".join(
            normalize_phone(phone)
            for phone in str(row.phones).split()
        )
        lexicon[word][phones] += int(row.count)

    return lexicon


def is_float(text: str) -> bool:
    """Проверить, является ли строка числом"""

    try:
        float(text)
        return True
    except ValueError:
        return False


def load_mfa_lexicon(
    path: Path = MFA_LEXICON_PATH,
    allowed_phones: set[str] | None = None,
) -> dict[str, Counter]:
    """Прочитать Russian MFA dictionary

    В release-версии перед транскрипцией могут находиться несколько чисел
    Поэтому здесь отдельно проверяю формат и беру фонемы из последнего столбца
    """

    lexicon: dict[str, Counter] = defaultdict(Counter)

    if not path.exists():
        print(f"Внимание: {path} не найден")
        return lexicon

    skipped = 0
    variants = 0

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            columns = line.split("\t")

            if len(columns) >= 2:
                word = normalize_word(columns[0])
                pronunciation = columns[-1].strip()
                probability = 1.0

                # Первое число после слова использую как вес варианта произношения
                if len(columns) >= 3 and is_float(columns[1].strip()):
                    probability = float(columns[1].strip())
            else:
                # Оставляю запасной вариант на случай словаря с обычными пробелами
                parts = line.split()
                if len(parts) < 2:
                    continue

                word = normalize_word(parts[0])
                rest = parts[1:]
                probability = 1.0

                if rest and is_float(rest[0]):
                    probability = float(rest[0])

                # Числа в начале строки не являются фонемами
                while rest and is_float(rest[0]):
                    rest.pop(0)

                pronunciation = " ".join(rest)

            phones = [
                normalize_phone(phone)
                for phone in pronunciation.split()
            ]

            if not phones or "spn" in phones:
                continue

            # Чтобы не выпустить обозначение, которого нет в наборе
            if allowed_phones is not None and any(
                phone not in allowed_phones
                for phone in phones
            ):
                skipped += 1
                continue

            lexicon[word][" ".join(phones)] += probability
            variants += 1

    print(
        f"Russian MFA: загружено {len(lexicon)} слов / "
        f"{variants} вариантов"
    )

    if skipped:
        print(
            "Пропущено вариантов с фонемами "
            f"вне phones_mfa.txt: {skipped}"
        )

    return lexicon


def best_pronunciation(variants: Counter) -> list[str]:
    """Выбрать самый частый или самый вероятный вариант произношения"""

    if not variants:
        return []

    phones, _ = variants.most_common(1)[0]
    return phones.split()


class StressPredictor:
    """Простая оценка места ударения по train-словарю

    Отдельного знака ударения в токенах нет
    Поэтому сначала пытаюсь найти похожее начало слова, а затем похожее окончание
    """

    def __init__(self, corpus_lexicon: dict[str, Counter]) -> None:
        self.suffix_stats = {
            n: defaultdict(Counter)
            for n in range(1, 7)
        }
        self.stem_stats: dict[str, Counter] = defaultdict(Counter)
        self.global_stats = Counter()

        self._fit(corpus_lexicon)

    @staticmethod
    def _vowel_letters(word: str) -> list[str]:
        return [
            char
            for char in word
            if char in VOWEL_LETTERS
        ]

    @staticmethod
    def _vowel_phones(phones: list[str]) -> list[str]:
        return [
            phone
            for phone in phones
            if phone in VOWEL_PHONES
        ]

    def _find_stress(
        self,
        word: str,
        phones: list[str],
    ) -> int | None:
        """Попробовать определить номер ударной гласной от начала слова"""

        letters = self._vowel_letters(word)
        vowels = self._vowel_phones(phones)

        if not letters:
            return None

        # Если гласная одна, выбирать место ударения не нужно
        if len(letters) == 1:
            return 0

        # Для ё считаю ударение известным сразу
        if "ё" in letters:
            return letters.index("ё")

        # Неоднозначные примеры пропускаю
        if len(letters) != len(vowels):
            return None

        candidates = [
            i
            for i, phone in enumerate(vowels)
            if phone in STRESSED_VOWELS and phone != "ɨ"
        ]

        if len(candidates) != 1:
            return None

        return candidates[0]

    def _fit(self, corpus_lexicon: dict[str, Counter]) -> None:
        """Собрать простую статистику только по train-словарю"""

        for word, variants in corpus_lexicon.items():
            phones = best_pronunciation(variants)
            stress_index = self._find_stress(word, phones)

            if stress_index is None:
                continue

            vowel_count = len(self._vowel_letters(word))
            from_end = vowel_count - 1 - stress_index
            self.global_stats[from_end] += 1

            # По окончаниям запоминаю положение ударения от конца слова
            for n in range(1, min(6, len(word)) + 1):
                ending = word[-n:]
                self.suffix_stats[n][ending][from_end] += 1

            # По началу слова запоминаю положение ударения от начала
            # Это помогает для похожих слов и словоформ
            for cut in range(0, min(6, max(0, len(word) - 4)) + 1):
                stem = word[:-cut] if cut else word
                if len(stem) >= 4:
                    self.stem_stats[stem][stress_index] += 1

    def predict_from_end(self, word: str) -> int:
        """Вернуть номер ударной гласной, считая от конца слова"""

        vowels = self._vowel_letters(word)

        if not vowels:
            return 0

        if "ё" in vowels:
            return len(vowels) - 1 - vowels.index("ё")

        # Сначала пробую найти знакомое начало слова
        for cut in range(1, min(7, max(1, len(word) - 3))):
            stem = word[:-cut]
            stats = self.stem_stats.get(stem)

            if not stats:
                continue

            stress_from_start = stats.most_common(1)[0][0]
            if stress_from_start < len(vowels):
                return len(vowels) - 1 - stress_from_start

        # Если похожего начала нет, смотрю самое длинное знакомое окончание
        for n in range(min(6, len(word)), 0, -1):
            ending = word[-n:]
            stats = self.suffix_stats[n].get(ending)

            # Очень редкое окончание может случайно дать плохую подсказку
            if stats and sum(stats.values()) >= 3:
                return stats.most_common(1)[0][0]

        if self.global_stats:
            return self.global_stats.most_common(1)[0][0]

        return 0


class OOVTranscriber:
    """Транскрибировать слова, которых нет ни в одном словаре"""

    # Для согласных храню твердый и мягкий варианты
    # Так проще обрабатывать смягчение
    CONSONANTS = {
        "б": ("b", "bʲ"),
        "в": ("v", "vʲ"),
        "г": ("ɡ", "ɟ"),
        "д": ("d̪", "dʲ"),
        "ж": ("ʐ", "ʐ"),
        "з": ("z̪", "zʲ"),
        "к": ("k", "c"),
        "л": ("ɫ", "ʎ"),
        "м": ("m", "mʲ"),
        "н": ("n̪", "ɲ"),
        "п": ("p", "pʲ"),
        "р": ("r", "rʲ"),
        "с": ("s̪", "sʲ"),
        "т": ("t̪", "tʲ"),
        "ф": ("f", "fʲ"),
        "х": ("x", "ç"),
        "ц": ("t̪s̪", "t̪s̪"),
        "ч": ("tɕ", "tɕ"),
        "ш": ("ʂ", "ʂ"),
        "щ": ("ɕ", "ɕ"),
        "й": ("j", "j"),
    }

    SIMPLE_MAP = {
        "а": "a",
        "б": "b",
        "в": "v",
        "г": "ɡ",
        "д": "d̪",
        "е": "e",
        "ё": "o",
        "ж": "ʐ",
        "з": "z̪",
        "и": "i",
        "й": "j",
        "к": "k",
        "л": "ɫ",
        "м": "m",
        "н": "n̪",
        "о": "o",
        "п": "p",
        "р": "r",
        "с": "s̪",
        "т": "t̪",
        "у": "u",
        "ф": "f",
        "х": "x",
        "ц": "t̪s̪",
        "ч": "tɕ",
        "ш": "ʂ",
        "щ": "ɕ",
        "ы": "ɨ",
        "э": "ɛ",
        "ю": "u",
        "я": "a",
    }

    VOICED = set("бвгджз")
    VOICE_TRIGGERS = set("бгджз")
    VOICELESS = set("пфктшсхцчщ")

    VOICE_PAIRS = {
        "б": "п",
        "п": "б",
        "в": "ф",
        "ф": "в",
        "г": "к",
        "к": "г",
        "д": "т",
        "т": "д",
        "ж": "ш",
        "ш": "ж",
        "з": "с",
        "с": "з",
    }

    SOFT_VOWELS = set("еёиюя")

    def __init__(
        self,
        corpus_lexicon: dict[str, Counter],
        allowed_phones: set[str] | None = None,
    ) -> None:
        self.allowed_phones = allowed_phones
        self.stress = StressPredictor(corpus_lexicon)

    def naive(self, word: str) -> list[str]:
        """Самый простой baseline, где буква почти напрямую переходит в фонему"""

        result = []

        for char in normalize_word(word):
            if char in {"ь", "ъ"}:
                continue

            phone = self.SIMPLE_MAP.get(char)
            if phone is not None:
                result.append(phone)

        return result

    def _stress_positions(self, word: str) -> tuple[int, int]:
        """Найти позиции ударной и предударной гласной"""

        vowel_positions = [
            i
            for i, char in enumerate(word)
            if char in VOWEL_LETTERS
        ]

        if not vowel_positions:
            return -1, -1

        from_end = self.stress.predict_from_end(word)
        stress_number = len(vowel_positions) - 1 - from_end
        stress_number = max(0, min(stress_number, len(vowel_positions) - 1))

        stress_position = vowel_positions[stress_number]
        before_position = (
            vowel_positions[stress_number - 1]
            if stress_number > 0
            else -1
        )

        return stress_position, before_position

    def _consonant(
        self,
        word: str,
        i: int,
        final_rules: bool,
    ) -> str | None:
        """Получить фонему для одной согласной"""

        char = word[i]
        n = len(word)

        if char not in self.CONSONANTS:
            return None

        # В двойной согласной оставляю один звук
        # Беру вторую букву пары, потому что она видит следующую гласную
        if final_rules and i + 1 < n and word[i + 1] == char:
            return None

        # В сочетании тч/дч первый звук отдельно не добавляю
        if (
            final_rules
            and char in {"т", "д"}
            and i + 1 < n
            and word[i + 1] == "ч"
        ):
            return None

        actual = char

        if final_rules:
            j = i + 1

            # Ь и Ъ сами звука не дают, поэтому смотрю через них
            while j < n and word[j] in {"ь", "ъ"}:
                j += 1

            next_char = word[j] if j < n else ""

            # Звонкая согласная может оглушаться перед глухой и в конце слова
            if char in self.VOICED and (
                j >= n
                or next_char in self.VOICELESS
            ):
                actual = self.VOICE_PAIRS.get(char, char)

            # Глухая согласная может озвончаться перед звонкой
            # Букву в здесь специально не считаю причиной озвончения
            elif char in self.VOICELESS and next_char in self.VOICE_TRIGGERS:
                actual = self.VOICE_PAIRS.get(char, char)

        next_char = word[i + 1] if i + 1 < n else ""

        soft = (
            (next_char in self.SOFT_VOWELS or next_char == "ь")
            and char not in {"ж", "ш", "ц"}
        )

        # Н перед мягкими т/д тоже часто становится мягкой
        if (
            final_rules
            and char == "н"
            and i + 2 < n
            and word[i + 1] in {"т", "д"}
            and word[i + 2] in (self.SOFT_VOWELS | {"ь"})
        ):
            soft = True

        hard_phone, soft_phone = self.CONSONANTS.get(
            actual,
            self.CONSONANTS[char],
        )

        return soft_phone if soft else hard_phone

    def _vowel(
        self,
        word: str,
        i: int,
        stress_position: int,
        before_position: int,
        use_stress: bool,
    ) -> list[str]:
        """Получить одну или две фонемы для гласной буквы"""

        char = word[i]
        result = []

        # Е/Ё/Ю/Я в начале слова, после гласной, Ь или Ъ могут добавлять j
        if (
            char in {"е", "ё", "ю", "я"}
            and (
                i == 0
                or word[i - 1] in VOWEL_LETTERS
                or word[i - 1] in {"ь", "ъ"}
            )
        ):
            result.append("j")

        if not use_stress:
            simple_vowels = {
                "а": "a",
                "о": "o",
                "э": "ɛ",
                "е": "e",
                "и": "i",
                "ы": "ɨ",
                "у": "u",
                "ё": "o",
                "я": "a",
                "ю": "u",
            }
            result.append(simple_vowels[char])
            return result

        stressed = i == stress_position
        before_stress = i == before_position

        previous_is_soft = (
            i > 0
            and word[i - 1] in self.CONSONANTS
            and char in self.SOFT_VOWELS
            and word[i - 1] not in {"ж", "ш", "ц"}
        )

        # Ниже правила отдельно по буквам
        # Так проще проверять, откуда получилась конкретная фонема
        if char == "а":
            if stressed and previous_is_soft:
                phone = "æ"
            elif stressed:
                phone = "a"
            elif before_stress or i == 0:
                phone = "ɐ"
            else:
                phone = "ə"

        elif char == "о":
            if stressed:
                phone = "o"
            elif before_stress or i == 0:
                phone = "ɐ"
            else:
                phone = "ə"

        elif char == "э":
            phone = "ɛ" if stressed else "ɪ"

        elif char == "е":
            phone = "e" if stressed else "ɪ"

        elif char == "и":
            if stressed:
                phone = "i"
            elif i > 0 and word[i - 1] in {"ж", "ш", "ц"}:
                phone = "ɨ"
            else:
                phone = "ɪ"

        elif char == "ы":
            phone = "ɨ"

        elif char == "у":
            if stressed and previous_is_soft:
                phone = "ʉ"
            elif stressed:
                phone = "u"
            else:
                phone = "ʊ"

        elif char == "ё":
            phone = "ɵ" if previous_is_soft else "o"

        elif char == "я":
            if stressed and i > 0 and word[i - 1] in self.CONSONANTS:
                phone = "æ"
            elif stressed:
                phone = "a"
            else:
                phone = "ɪ"

        elif char == "ю":
            if stressed and i > 0 and word[i - 1] in self.CONSONANTS:
                phone = "ʉ"
            elif stressed:
                phone = "u"
            else:
                phone = "ʊ"

        else:
            return result

        result.append(phone)
        return result

    def transcribe(self, word: str, mode: str = "final") -> list[str]:
        """Транскрибировать одно OOV-слово выбранным набором правил"""

        word = normalize_word(word)

        if mode == "naive":
            return self.naive(word)

        if mode not in {"soft", "stress", "final"}:
            raise ValueError(f"Unknown OOV mode: {mode}")

        use_stress = mode in {"stress", "final"}
        final_rules = mode == "final"

        stress_position, before_position = (
            self._stress_positions(word)
            if use_stress
            else (-1, -1)
        )

        result = []

        for i, char in enumerate(word):
            if char in {"ь", "ъ"}:
                continue

            if char in self.CONSONANTS:
                phone = self._consonant(
                    word,
                    i,
                    final_rules=final_rules,
                )
                if phone is not None:
                    result.append(phone)
                continue

            if char in VOWEL_LETTERS:
                result.extend(
                    self._vowel(
                        word,
                        i,
                        stress_position,
                        before_position,
                        use_stress=use_stress,
                    )
                )

        if self.allowed_phones is not None:
            result = [
                phone
                for phone in result
                if phone in self.allowed_phones
            ]

        return result


class G2P:
    """Преобразовать последовательность токенов в последовательности фонем

    На вход приходит одно предложение без пунктуации
    Внутри него может находиться <SIL>, который нужно вернуть без изменений
    """

    def __init__(
        self,
        corpus_lexicon: dict[str, Counter],
        mfa_lexicon: dict[str, Counter] | None = None,
        allowed_phones: set[str] | None = None,
        oov_mode: str = "naive",
    ) -> None:
        self.corpus_lexicon = corpus_lexicon
        self.mfa_lexicon = mfa_lexicon or {}
        self.allowed_phones = allowed_phones
        self.oov_mode = oov_mode

        self.oov = OOVTranscriber(
            corpus_lexicon,
            allowed_phones=allowed_phones,
        )

    def source(self, token: str) -> str:
        """Показать, откуда было взято произношение токена"""

        if token == SIL:
            return "sil"

        word = normalize_word(token)

        if word in self.corpus_lexicon:
            return "corpus"

        if word in self.mfa_lexicon:
            return "mfa"

        return "oov"

    def transcribe(
        self,
        tokens: list[str] | np.ndarray,
    ) -> list[list[str]]:
        """Транскрибировать токены одного предложения в том же порядке"""

        result = []

        for token in tokens:
            token = str(token)

            if token == SIL:
                result.append([SIL])
                continue

            word = normalize_word(token)

            if word in self.corpus_lexicon:
                phones = best_pronunciation(
                    self.corpus_lexicon[word]
                )
            elif word in self.mfa_lexicon:
                phones = best_pronunciation(
                    self.mfa_lexicon[word]
                )
            else:
                phones = self.oov.transcribe(
                    word,
                    mode=self.oov_mode,
                )

            result.append(phones)

        return result


def edit_distance(
    reference: list[str],
    hypothesis: list[str],
) -> int:
    """Посчитать расстояние Левенштейна между двумя списками фонем"""

    previous = list(range(len(hypothesis) + 1))

    for i, ref_phone in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)

        for j, hyp_phone in enumerate(hypothesis, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_phone != hyp_phone),
            )

        previous = current

    return previous[-1]


def phone_error_rate(
    reference: list[list[str]],
    hypothesis: list[list[str]],
) -> float:
    """Посчитать PER как сумму ошибок, делённую на число эталонных фонем"""

    total_distance = 0
    total_length = 0

    for ref, hyp in zip(reference, hypothesis):
        total_distance += edit_distance(ref, hyp)
        total_length += len(ref)

    if total_length == 0:
        return float("nan")

    return total_distance / total_length


def word_accuracy(
    reference: list[list[str]],
    hypothesis: list[list[str]],
) -> float:
    """Посчитать долю слов с полностью совпавшей транскрипцией"""

    if not reference:
        return float("nan")

    return float(
        np.mean([
            ref == hyp
            for ref, hyp in zip(reference, hypothesis)
        ])
    )


def calc_metrics(
    g2p: G2P,
    tokens: pd.DataFrame,
    title: str = "test",
    verbose: bool = True,
) -> dict[str, float]:
    """Посчитать метрики отдельно для словаря корпуса, MFA и остатка OOV"""

    sentence_ref = []
    sentence_hyp = []

    refs = {
        "corpus": [],
        "mfa": [],
        "oov": [],
    }
    hyps = {
        "corpus": [],
        "mfa": [],
        "oov": [],
    }

    groups = tokens.groupby("id", sort=False)

    if verbose:
        groups = tqdm(groups, desc=f"Оценка {title}")

    for _, sentence in groups:
        sentence = sentence.sort_values("token_id")
        words = sentence["token"].values
        predicted = g2p.transcribe(words)

        if len(predicted) != len(words):
            raise ValueError(
                "G2P вернул неправильное число результатов"
            )

        reference = [
            [
                normalize_phone(phone)
                for phone in str(phones).split()
                if phone not in {SIL, "spn"}
            ]
            for phones in sentence["phones"].values
        ]

        # <SIL> не включаю в PER, т.к. места пауз оценивались в прошлой работе
        sentence_ref.append([
            phone
            for word in reference
            for phone in word
            if phone != SIL
        ])
        sentence_hyp.append([
            phone
            for word in predicted
            for phone in word
            if phone != SIL
        ])

        # Отдельные списки нужны, чтобы увидеть, где именно появляется ошибка
        for token, ref, hyp in zip(words, reference, predicted):
            source = g2p.source(str(token))

            if source == "sil":
                continue

            refs[source].append(ref)
            hyps[source].append(hyp)

    result = {
        "running_per": phone_error_rate(
            sentence_ref,
            sentence_hyp,
        ),
    }

    for source in ["corpus", "mfa", "oov"]:
        result[f"{source}_per"] = phone_error_rate(
            refs[source],
            hyps[source],
        )
        result[f"{source}_accuracy"] = word_accuracy(
            refs[source],
            hyps[source],
        )
        result[f"{source}_words"] = len(refs[source])

    if verbose:
        print()
        print(title)
        print("=" * len(title))
        print(
            "PER на всём тексте: "
            f"{result['running_per']:.4f}"
        )
        print(
            "lexicon.csv: "
            f"{result['corpus_words']} слов, "
            f"PER={result['corpus_per']:.4f}"
        )
        print(
            "Russian MFA: "
            f"{result['mfa_words']} слов, "
            f"PER={result['mfa_per']:.4f}"
        )
        print(
            "Остаток OOV: "
            f"{result['oov_words']} слов, "
            f"PER={result['oov_per']:.4f}, "
            f"accuracy={result['oov_accuracy']:.4f}"
        )

    return result


def collect_oov_errors(
    g2p: G2P,
    tokens: pd.DataFrame,
) -> pd.DataFrame:
    """Собрать отдельную таблицу ошибок только для residual OOV"""

    rows = []

    for row in tokens.itertuples(index=False):
        word = str(row.token)

        if g2p.source(word) != "oov":
            continue

        reference = [
            normalize_phone(phone)
            for phone in str(row.phones).split()
            if phone not in {SIL, "spn"}
        ]
        predicted = g2p.transcribe([word])[0]
        distance = edit_distance(reference, predicted)

        rows.append({
            "id": row.id,
            "token": word,
            "reference": " ".join(reference),
            "predicted": " ".join(predicted),
            "edits": distance,
            "phones": len(reference),
            "word_per": (
                distance / len(reference)
                if reference
                else math.nan
            ),
            "correct": reference == predicted,
        })

    return pd.DataFrame(rows)


def check_inventory(
    g2p: G2P,
    tokens: pd.DataFrame,
) -> set[str]:
    """Проверить, не появились ли фонемы вне phones_mfa.txt"""

    if g2p.allowed_phones is None:
        return set()

    predicted_phones = set()

    for _, sentence in tokens.groupby("id", sort=False):
        words = (
            sentence
            .sort_values("token_id")["token"]
            .values
        )
        predicted = g2p.transcribe(words)

        for word in predicted:
            predicted_phones.update(word)

    return predicted_phones - g2p.allowed_phones


def create_g2p(oov_mode: str = "final") -> G2P:
    """Создать G2P с теми же словарями, что используются при обычном запуске"""

    allowed = load_allowed_phones()
    corpus = load_corpus_lexicon()
    mfa = load_mfa_lexicon(
        allowed_phones=allowed,
    )

    return G2P(
        corpus_lexicon=corpus,
        mfa_lexicon=mfa,
        allowed_phones=allowed,
        oov_mode=oov_mode,
    )


def main() -> None:
    """Сравнить baseline и все добавленные улучшения"""

    tokens = load_tokens()
    train = tokens[tokens["set"] == "train"].copy()
    test = tokens[tokens["set"] == "test"].copy()

    allowed = load_allowed_phones()
    corpus = load_corpus_lexicon()
    mfa = load_mfa_lexicon(
        allowed_phones=allowed,
    )

    modes = [
        ("Baseline", "naive"),
        ("+ смягчение", "soft"),
        ("+ ударение", "stress"),
        ("Финальный", "final"),
    ]

    rows = []

    for title, mode in modes:
        g2p = G2P(
            corpus_lexicon=corpus,
            mfa_lexicon=mfa,
            allowed_phones=allowed,
            oov_mode=mode,
        )
        metrics = calc_metrics(
            g2p,
            test,
            title=title,
            verbose=False,
        )

        rows.append({
            "approach": title,
            "test_per": metrics["running_per"],
            "oov_per": metrics["oov_per"],
            "oov_accuracy": metrics["oov_accuracy"],
        })

    comparison = pd.DataFrame(rows)

    print()
    print("Сравнение OOV")
    print("=============")
    print(comparison.to_string(index=False))

    final_g2p = G2P(
        corpus_lexicon=corpus,
        mfa_lexicon=mfa,
        allowed_phones=allowed,
        oov_mode="final",
    )

    print()
    calc_metrics(
        final_g2p,
        train,
        title="Final / Train",
    )

    final_metrics = calc_metrics(
        final_g2p,
        test,
        title="Final / Test",
    )

    unknown = check_inventory(
        final_g2p,
        test,
    )

    print()
    print("Проверка требований")
    print("===================")
    print(
        "Test PER <= 0.025:",
        final_metrics["running_per"] <= 0.025,
    )
    print(
        "OOV PER <= 0.20:",
        final_metrics["oov_per"] <= 0.20,
    )
    print(
        "Лишние фонемы:",
        sorted(unknown),
    )


if __name__ == "__main__":
    main()
