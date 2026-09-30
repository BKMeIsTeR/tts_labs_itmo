"""Normalized / non-normalized classifier — skeleton for lab 1.

Run as a script to score yourself on the development set::

    python text_filter.py
"""

import csv
import re
import unicodedata


import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

DEV_SET_PATH = "data/dev_sentences.csv"


class TextFilter:
    """Decides whether an utterance is usable as a training example.

    Example:
        >>> textfilter = TextFilter()
        >>> textfilter.filter("Я вышел из дома.")
        1
        >>> textfilter.filter("Александрову Г. П.")
        0
    """

    def __init__(self):
        """Prepare the classifier's resources.

        Compiled regular expressions, abbreviation and contraction dictionaries, a
        trained model — anything that should not be rebuilt for every utterance.
        """
        # НЕ считаем любое слово, написанное CAPS LOCK-ом, аббревиатурой
        # В RUSLAN много обычных заголовков:
        # "ПЕРВЫЙ КРИТИК", "СОЛО НА УНДЕРВУДЕ"
        #
        # Поэтому используем список реальных аббревиатур, найденных в корпусе
        abbrev_list = [
            "АН", "АТС", "АХЧ", "БТ", "ВМК", "ГБ", "ГПУ", "ДПИ",
            "КВВК", "КВН", "КП", "КПП", "КПСС", "КПЭ", "ЛГУ",
            "МВД", "МГУ", "МПВО", "МТС", "НКВД", "ПТУ", "СНО",
            "СС", "ССР", "СХШ", "США", "ТГУ", "ТПИ", "УВД",
            "ФБР", "ФД", "ЦДЛ", "ЦО", "ЧК",
            "БАМ", "БУР", "ВОХРА", "ГАИ", "ГУЛАГ", "ДОСААФ",
            "ЛИТМО", "ЛОМО", "ЛОСП", "ОВИР", "ОК", "ООН",
            "РОМ", "ШИЗО",
        ]

        abbrev_pattern = (
            r"\b(?:"
            + "|".join(re.escape(abbrev) for abbrev in abbrev_list)
            + r")\b"
        )

        self.patterns = {
            "abbreviation": re.compile(abbrev_pattern),

            # Сокращения вроде "проф.", "д-р"
            "other_abbreviation": re.compile(
                r"\b(?:тов\.|проф\.|доц\.|акад\.|д-р|канд\.)",
                re.IGNORECASE,
            ),

            # Сокращения с точками
            "with_dots": re.compile(
                r"(?:"
                r"\bт\.\s*е\.?"
                r"|\bт\.\s*к\.?"
                r"|\bт\.\s*д\.?"
                r"|\bт\.\s*п\.?"
                r"|\bи\.\s*т\.\s*д\.?"
                r"|\bи\.\s*т\.\s*п\.?"
                r"|\bт\.\s*о\.?"
                r")",
                re.IGNORECASE,
            ),

            # Адресные и похожие сокращения
            "address": re.compile(
                r"(?:"
                r"\bул\."
                r"|\bд\."
                r"|\bг\."
                r"|\bпр\."
                r"|\bпер\."
                r"|\bпл\."
                r"|\bбульв\."
                r"|\bнаб\."
                r"|\bш\."
                r"|\bстр\."
                r"|\bрис\."
                r"|\bсм\."
                r"|\bтабл\."
                r"|\bг-(?:н|ну|на|ном|не|жа|жи|жу|жей)\b"
                r")",
                re.IGNORECASE,
            ),

            # Инициалы:
            # И. Петров
            # И. И. Петров
            # Иванов И. И.
            # В.И.Ленин
            "initials": re.compile(
                r"(?:"
                r"\b[А-ЯЁ]\.\s*(?:[А-ЯЁ]\.\s*)?[А-ЯЁ][а-яё]+"
                r"|"
                r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.\s*(?:[А-ЯЁ]\.)?"
                r"|"
                r"\b[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+"
                r")"
            ),

            # Технические символы
            "technical_symbols": re.compile(
                r"[*\/@+<>{}\[\]\\|~`=^&]"
            ),

           # Обозначения, которые имеют отдельное произношение
            "symbols": re.compile(
                r"[%°$€£¥₽©®™№±√∫∑∞≈≠≤≥]"
            ),

            # Текстовые смайлы
            "text_emoji": re.compile(
                r"(?:"
                r"[:;=][-o*^]?[()\[\]{}<>DdpPрР/\\|3cCOo0]"
                r"|<3"
                r"|3<"
                r"|;p"
                r")",
                re.IGNORECASE,
            ),

            # Междометия
            "interjection": re.compile(
                r"\b(?:"
                r"ага|агу|ай|ах|ахаха|брр|"
                r"гм+|ммм?|мда|ого|ой|ох|охохо|"
                r"угу|ура|ух|ха|хе|хехе|хм+|хы|"
                r"ых|ыы+|э|эх|хо|хи|гы|гх|"
                r"фу|фи|тьфу|эй|эге|эхе"
                r")\b",
                re.IGNORECASE,
            ),

            # Лишний пробел перед пунктуацией: "Спасибо , что..."
            "space_before_punctuation": re.compile(
                r"\s+[,.!?;:]"
            ),

            # Нет пробела после запятой/двоеточия/точки с запятой:
            # "Простите,я..."
            "missing_space_after_punctuation": re.compile(
                r"[,;:](?=[А-ЯЁа-яё])"
            ),

            # Неправильная повторная пунктуация.
            #
            # "!!", "???", "..", "!."
            #
            # Но "?!" и "?!." НЕ ошибка
            "bad_punctuation": re.compile(
                r"(?:"
                r"!!+"
                r"|\?\?+"
                r"|(?<!\.)\.\.(?!\.)"
                r"|(?<!\?)!\."
                r")"
            ),

            # Скобки в dev используются в телефонах, смайлах
            # или как артефакты текста.
            "brackets": re.compile(
                r"[()\[\]{}]"
            ),

            # "Минюст" отдельно приведён в условии как особый случай.
            "minjust": re.compile(
                r"\bМинюст\w*\b",
                re.IGNORECASE,
            ),
        }

        # Разрешённые знаки пунктуации.
        #
        # U+0301 (знак ударения) здесь не указан:
        # его отдельно разрешаем в _contains_forbidden_unicode().
        self.allowed_punctuation = set(
            '.,!?;:…—-–‑‐"\'«»'
        )
        
    def _contains_forbidden_unicode(self, text: str) -> bool:
        """Check digits, non-Cyrillic letters and unsupported symbols."""

        for char in text:
            # Знак ударения нужно сохранить для next лаб-ых
            if char == "\u0301":
                continue

            if char == " ":
                continue

            # Числа считаем ненормализованным текстом
            if char.isdigit():
                return True

            # Разрешаем только кириллические буквы
            if char.isalpha():
                unicode_name = unicodedata.name(char, "")

                if "CYRILLIC" not in unicode_name:
                    return True

                continue

            category = unicodedata.category(char)

            # Другие combining marks здесь не ожидаем
            if category.startswith("M"):
                return True

            # Пунктуация разрешена только из небольшого списка
            if category.startswith("P"):
                if char not in self.allowed_punctuation:
                    return True

                continue

            # Всё остальное:
            # emoji, валюты, математические символы,
            # zero-width characters, необычные пробелы и т. п.
            return True

        return False

    @staticmethod
    def _quotes_are_balanced(text: str) -> bool:
        """Check the simplest cases of unmatched quotes."""

        if text.count("«") != text.count("»"):
            return False

        if text.count('"') % 2 != 0:
            return False

        return True

    def filter(self, text: str) -> int:
        """Classify a single utterance.

        Args:
            text: Utterance text, already passed through :class:`TextNormalizer`.

        Returns:
            ``1`` if the text is normalized and the utterance can be used for
            training; 
            ``0`` if it contains something the speaker pronounced
            differently from how it is written, and the utterance should be dropped.
        """

        if not isinstance(text, str):
            return 0

        if not text.strip():
            return 0

        if self._contains_forbidden_unicode(text):
            return 0

        if not self._quotes_are_balanced(text):
            return 0

        for pattern in self.patterns.values():
            if pattern.search(text):
                return 0

        return 1


if __name__ == "__main__":
    textfilter = TextFilter()

    dev_files = pd.read_csv(
        DEV_SET_PATH, sep="|", encoding="utf-8", quoting=csv.QUOTE_NONE, header=0
    )

    dev_files["predicted"] = dev_files["text"].apply(textfilter.filter)
    
    prc = precision_score(dev_files["is_normalized"], dev_files["predicted"])
    rec = recall_score(dev_files["is_normalized"], dev_files["predicted"])
    f1 = f1_score(dev_files["is_normalized"], dev_files["predicted"])
    print(f"F1 Score is {f1:.4f}, Precision is {prc:.4f}, Recall is {rec:.4f}")
