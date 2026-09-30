"""Russian text normalizer - skeleton for lab 1.

Brings corpus text into a form usable for training a speech synthesizer.
"""


import re
import unicodedata


class TextNormalizer:
    """Normalizes text in Russian.


        "!.."           -> "!"
        "«цитата»"      -> '"цитата"'
        "текст * мусор" -> "текст мусор"
        "де‑факто"      -> "де-факто"      # U+2011 -> ordinary hyphen

    **Word-changing edits.** The alignment for that utterance becomes invalid and the
    row must be dropped from the training set - but the logic itself is still needed
    for lab 5, where arbitrary user input arrives with no alignment at all::

        "в 1995 г."     -> "в тысяча девятьсот девяносто пятом году"
        "прим. автора"  -> "примечание автора"

    Example:
        >>> normalizer = TextNormalizer()
        >>> normalizer.normalize("Расстреливать надо таких писателей!.")
        'Расстреливать надо таких писателей!'
    """

    def __init__(self):
        """Prepare the normalizer's resources.

        Put anything expensive to build here: compiled regular expressions,
        abbreviation and contraction dictionaries, a morphological analyzer.
        Building them inside :meth:`normalize` means building them 22,200 times.
        """

        # Unicode-пробелы -> обычный пробел
        self.unicode_spaces = re.compile(
            r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]"
        )

        # Невидимые символы после copy/paste.
        self.zero_width_chars = re.compile(
            r"[\u200B\u200C\u200D\u2060\uFEFF]"
        )

        # Разные варианты кавычек -> обычные двойные / одинарные кавычки.
        self.quote_translation = str.maketrans({
            "«": '"',
            "»": '"',
            "“": '"',
            "”": '"',
            "„": '"',
            "‘": "'",
            "’": "'",
        })

        # Некоторые hyphen-like символы -> обычный дефис.
        #
        # Длинное тире и en dash оставляем как допустимую пунктуацию.
        self.hyphen_translation = str.maketrans({
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2212": "-",
        })

        # Из задания: "текст * мусор" -> "текст мусор"
        # Другие смысловые символы (% / @ ° ₽ ...) НЕ удаляем
        self.decorative_chars = re.compile(r"[*`()\[\]{}]")

        # Очевидные проблемы пунктуации
        self.repeated_exclamation = re.compile(r"!{2,}")
        self.repeated_question = re.compile(r"\?{2,}")
        self.mark_followed_by_dots = re.compile(r"([!?])\.+")
        self.double_dot = re.compile(r"(?<!\.)\.\.(?!\.)")
        self.repeated_soft_punctuation = re.compile(r"([,;:])\1+")

        # Пробелы около пунктуации
        self.space_before_punctuation = re.compile(r"\s+([,.;:!?])")
        self.missing_space_after_punctuation = re.compile(
            r"([,;:])(?=[А-ЯЁа-яё])"
        )

        self.multiple_spaces = re.compile(r"[ \t\r\n]+")

    def normalize(self, text: str) -> str:
        """Normalize a single line.

        Args:
            text: Raw utterance text, exactly as stored in the corpus metadata.

        Returns:
            The normalized text. Returning the input unchanged is valid and common -
            most lines need nothing done to them.

        Note:
            Do not strip the combining acute accent ``U+0301``. It looks like part of
            the letter and is easily lost to "unicode cleanup", but it marks explicit
            stress and becomes labelled data for stress placement in lab 3.

            Normalize to NFC. Strings in NFC and NFD render identically in a terminal
            and compare unequal.
        """

        if text is None:
            return ""

        text = str(text)

        # 1. Обязательная Unicode-нормализация
        #   Знак ударения U+0301 не удаляем
        text = unicodedata.normalize("NFC", text)

        # 2. Пробелы и невидимые символы
        text = self.unicode_spaces.sub(" ", text)
        text = self.zero_width_chars.sub("", text)

        # 3. Кавычки и дефисы
        text = text.translate(self.quote_translation)
        text = text.translate(self.hyphen_translation)

        # 4. Только явно декоративные символы
        text = self.decorative_chars.sub("", text)

        # 5. Повторяющаяся / сломанная пунктуация
        text = self.repeated_exclamation.sub("!", text)
        text = self.repeated_question.sub("?", text)

        # "!.." -> "!", "?.." -> "?"
        text = self.mark_followed_by_dots.sub(r"\1", text)

        # Ровно две точки считаем опечаткой
        # Три точки "..." сохраняем
        text = self.double_dot.sub(".", text)

        text = self.repeated_soft_punctuation.sub(r"\1", text)

        # 6. Исправляем пробелы около пунктуации
        text = self.space_before_punctuation.sub(r"\1", text)
        text = self.missing_space_after_punctuation.sub(r"\1 ", text)

        # 7. Убираем повторные пробелы
        text = self.multiple_spaces.sub(" ", text).strip()

        # 8. Если после унификации кавычек осталась непарная кавычка
        #   удаляем только знак кавычки
        if text.count('"') % 2 != 0:
            text = text.replace('"', "")

        # После удаления скобок/кавычек ещё раз нормализуем пробелы
        text = self.multiple_spaces.sub(" ", text).strip()

        # 9. На всякий случай ещё раз приводим результат к NFC
        return unicodedata.normalize("NFC", text)

if __name__ == "__main__":
    normalizer = TextNormalizer()

    examples = [
        "Расстреливать надо таких писателей!.",
        "«цитата»",
        "текст * мусор",
        "де‑факто",
        "Спасибо , что дождались ответа.",
    ]

    for example in examples:
        print(f"{example!r} -> {normalizer.normalize(example)!r}")