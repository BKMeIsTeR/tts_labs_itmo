"""Build the filtered RUSLAN metadata — entry point for lab 1.

Reads the corpus metadata (two columns), runs every utterance through the normalizer,
drops whatever the classifier rejects, and writes the result in LJSpeech format
(three columns)::

    000000_RUSLAN|С тревожным чувством берусь я за перо.|С тревожным чувством берусь я за перо.
                 ^ raw text                             ^ normalized text

The third column is your contribution: the original corpus does not have one. For most
rows it will equal the second, and that is expected — punctuation cleanup changes little.

Run from the lab directory::

    python preprocess_ruslan.py
"""

import csv

import pandas as pd

from text_filter import TextFilter
from text_normalizer import TextNormalizer

INPUT_PATH = "../../data/RUSLAN/metadata_RUSLAN_22200.csv"
OUTPUT_PATH = "../../data/metadata_RUSLAN_22200_normalized.csv"

# quoting=csv.QUOTE_NONE is required in both directions: the corpus text contains
# « » „ “ ” ' and pandas would otherwise read them as field delimiters.
CSV_KWARGS = {"sep": "|", "quoting": csv.QUOTE_NONE}


if __name__ == "__main__":
    text_filter = TextFilter()
    normalizer = TextNormalizer()

    raw = pd.read_csv(INPUT_PATH, names=["id", "raw"], **CSV_KWARGS)

    # Оценим корпус до нормализации.
    raw["before_ok"] = raw["raw"].apply(text_filter.filter)

    # Сначала normalizer, затем filter — как требуется в лабораторной.
    raw["nrm"] = raw["raw"].apply(normalizer.normalize)
    raw["after_ok"] = raw["nrm"].apply(text_filter.filter)

    before_normalized = int(raw["before_ok"].sum())
    after_normalized = int(raw["after_ok"].sum())
    rescued = int(
        ((raw["before_ok"] == 0) & (raw["after_ok"] == 1)).sum()
    )

    total = len(raw)

    print(f"Загружено {total} записей")
    print()
    print("До нормализации:")
    print(
        f"  Нормализованных: {before_normalized} "
        f"({before_normalized / total * 100:.2f}%)"
    )
    print(
        f"  Ненормализованных: {total - before_normalized} "
        f"({(total - before_normalized) / total * 100:.2f}%)"
    )
    print()
    print("После нормализации:")
    print(
        f"  Нормализованных: {after_normalized} "
        f"({after_normalized / total * 100:.2f}%)"
    )
    print(
        f"  Ненормализованных: {total - after_normalized} "
        f"({(total - after_normalized) / total * 100:.2f}%)"
    )
    print()
    print(
        "Удалось исправить для прохождения фильтра: "
        f"{rescued} предложений"
    )
    print(
        "Остались ненормализованными: "
        f"{total - after_normalized} предложений"
    )

    clean = raw[raw["after_ok"] == 1]

    clean[["id", "raw", "nrm"]].to_csv(
        OUTPUT_PATH,
        index=False,
        header=False,
        **CSV_KWARGS,
    )