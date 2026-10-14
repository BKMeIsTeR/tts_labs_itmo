"""Build the lab 3 training data from the MFA alignment — lab 3.

Reads the TextGrids downloaded in lab 2 and writes two files.

`data/tokens_phones.csv` — **the training table**, one row per aligned token, in sentence
order. Same shape as lab 2's `RUSLAN_pause_metadata.csv`, so the two labs read alike::

    id|token_id|token|phones|set

`data/lexicon.csv` — the pronunciation dictionary, aggregated over the training fold::

    word|phones|count

The training table is the primary artifact: it keeps the sentence grouping, so a
word-level model, a sentence-level model and anything contextual can all be built from it.
`lexicon.csv` is a convenience summary for lookup and for word-level training, and it
necessarily throws the context away.

Run from the lab directory::

    python prepare_lexicon.py
"""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path
import pandas as pd
import unicodedata
from collections import Counter, defaultdict

from praatio import textgrid
from tqdm import tqdm


ALIGN_DIR = Path("../../data/RUSLAN_align_v2")
NORMALIZED_META = Path("../../data/metadata_RUSLAN_22200_normalized.csv")
TOKENS_PATH = Path("data/tokens_phones.csv")
TOKENS_SIL_PATH = Path("data/tokens_phones_sil.csv")
LEXICON_PATH = Path("data/lexicon.csv")

SIL = "<SIL>"
LENGTH_MARK = "ː"
TEST_EVERY = 5  # lab 2's split rule: test iff the utterance number divides by 5


def normalize_phone(phone: str) -> str:
    """Привести обозначение фонемы к тому же виду, что используется дальше"""

    phone = unicodedata.normalize("NFC", str(phone))
    return phone.replace(LENGTH_MARK, "")


def read_kept_ids(path: Path) -> set[str] | None:
    """Utterance ids that survived the lab 1 filter, or None if the file is absent.

    The alignment was built on **raw** text, so it contains utterances a normalizer would
    reject — including three all-caps abbreviations whose transcriptions `mfa g2p` invented
    (`кпсс` → `k r ə f ɨ s̪`). Restricting to the lab 1 output removes them.
    """

    if not path.exists():
        print(f"{path} не найден")
        print("Будут использованы все TextGrid")
        return None

    kept = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                kept.add(line.split("|", 1)[0])

    return kept


def phones_of_token(phones: list[tuple[float, float, str]],
                    start: float, end: float) -> list[str]:
    """Collect the phones belonging to one word interval.

    Membership is decided by the phone's midpoint rather than its boundaries: word and
    phone tiers are written independently and a shared boundary can differ in the last
    decimal.

    Args:
        phones: `(start, end, label)` of every phone interval of the utterance.
        start: Word interval start, seconds.
        end: Word interval end, seconds.

    Returns:
        Phone labels in time order, NFC-normalized with the length mark collapsed to match
        `configs/phones_mfa.txt`. An empty label (silence) becomes ``"<SIL>"``.
    """
    out = []
    for phone_start, phone_end, label in phones:
        if start <= (phone_start + phone_end) / 2 < end:
            label = unicodedata.normalize('NFC', label) if label else SIL
            out.append(label.replace(LENGTH_MARK, ''))
    return out


def read_one_textgrid(
    path: Path,
    keep_sil: bool,
) -> tuple[list[dict], bool]:
    """Прочитать один TextGrid и вернуть строки будущей таблицы"""

    utterance_id = path.stem

    try:
        tg = textgrid.openTextgrid(
            str(path),
            includeEmptyIntervals=True,
        )
        word_tier = tg.getTier("words")
        phone_tier = tg.getTier("phones")
    except Exception as exc:
        print(f"Не удалось прочитать {utterance_id}: {exc}")
        return [], False

    phone_entries = [
        (entry.start, entry.end, entry.label)
        for entry in phone_tier.entries
    ]

    # Использую то же правило train/test, что и в предыдущей лабораторной
    fold = (
        "test"
        if int(utterance_id.split("_")[0]) % TEST_EVERY == 0
        else "train"
    )

    rows = []
    token_id = 0
    has_spn = False

    for entry in word_tier.entries:
        # Пустой интервал words означает паузу
        if entry.label == "":
            if not keep_sil:
                continue

            rows.append({
                "id": utterance_id,
                "token_id": token_id,
                "token": SIL,
                "phones": SIL,
                "set": fold,
            })
            token_id += 1
            continue

        phones = phones_of_token(
            phone_entries,
            entry.start,
            entry.end,
        )

        if "spn" in phones:
            has_spn = True

        token = unicodedata.normalize(
            "NFC",
            str(entry.label),
        ).lower()

        rows.append({
            "id": utterance_id,
            "token_id": token_id,
            "token": token,
            "phones": " ".join(phones),
            "set": fold,
        })
        token_id += 1

    return rows, has_spn


def read_alignment(
    align_root: Path,
    keep: set[str] | None,
    keep_sil: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """Read every TextGrid in `align_root` into per-token records.

    Args:
        align_root: Directory with `{id}.TextGrid` files.
        keep: Utterance ids to keep, or None for all.
        keep_sil: retains `<SIL>` tokens if true, else drops them

    Returns:
        One dict per aligned token with keys `id`, `token_id`, `token`, `phones`, `set`.
    """

    paths = sorted(align_root.glob("*.TextGrid"))

    if not paths:
        raise FileNotFoundError(
            f"В {align_root} не найдено ни одного .TextGrid"
        )

    rows = []
    stats = {
        "all_textgrids": len(paths),
        "not_in_normalized": 0,
        "failed": 0,
        "spn_utterances": 0,
        "kept_utterances": 0,
    }

    for path in tqdm(paths, desc="Читаем TextGrid"):
        utterance_id = path.stem

        if keep is not None and utterance_id not in keep:
            stats["not_in_normalized"] += 1
            continue

        utterance_rows, has_spn = read_one_textgrid(
            path,
            keep_sil=keep_sil,
        )

        if not utterance_rows:
            stats["failed"] += 1
            continue

        # spn не даёт нормальной фонемной транскрипции
        # Поэтому по условию убираю всё предложение целиком
        if has_spn:
            stats["spn_utterances"] += 1
            continue

        rows.extend(utterance_rows)
        stats["kept_utterances"] += 1

    return rows, stats


def build_lexicon(rows: list[dict]) -> dict[str, Counter]:
    """Count pronunciations per token over the training fold only.

    A `spn` phone marks a word the aligner had no pronunciation for: the interval has a
    duration but no real transcription, so the row carries no lexical information.

    Building this over every fold instead would put the test words in the dictionary and
    make the reported PER meaningless.
    """
    lexicon: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        # Test здесь использовать нельзя, иначе его слова заранее попадут в словарь
        if row["set"] != "train":
            continue

        word = row["token"]
        phones = row["phones"].split()

        if word == SIL or not phones or "spn" in phones:
            continue

        lexicon[word][row["phones"]] += 1

    return lexicon


def print_stats(
    rows: list[dict],
    stats: dict[str, int],
    title: str,
) -> None:
    """Напечатать короткую проверку подготовленных данных"""

    df = pd.DataFrame(rows)

    print()
    print(title)
    print("-" * len(title))
    print(f"TextGrid всего: {stats['all_textgrids']}")
    print(f"Не было после нормализации: {stats['not_in_normalized']}")
    print(f"Не удалось прочитать: {stats['failed']}")
    print(f"Удалено из-за spn: {stats['spn_utterances']}")
    print(f"Оставлено предложений: {stats['kept_utterances']}")
    print(f"Строк в таблице: {len(df)}")

    if len(df):
        print()
        print(df["set"].value_counts())


def main() -> None:
    """Read the alignment, write the training table and the lexicon."""
    Path("data").mkdir(exist_ok=True)
    kept_ids = read_kept_ids(NORMALIZED_META)

    # Эта версия нужна для обучения и оценки G2P
    rows, stats = read_alignment(
        ALIGN_DIR,
        kept_ids,
        keep_sil=False,
    )

    # Эта версия нужна дальше, потому что в ней остаются паузы
    rows_sil, stats_sil = read_alignment(
        ALIGN_DIR,
        kept_ids,
        keep_sil=True,
    )

    if not rows:
        raise RuntimeError("После обработки не осталось данных")

    tokens_df = pd.DataFrame(rows)
    tokens_sil_df = pd.DataFrame(rows_sil)

    tokens_df.to_csv(
        TOKENS_PATH,
        sep="|",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    tokens_sil_df.to_csv(
        TOKENS_SIL_PATH,
        sep="|",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    lexicon = build_lexicon(rows)

    with LEXICON_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(
            file,
            delimiter="|",
            quoting=csv.QUOTE_NONE,
        )
        writer.writerow(["word", "phones", "count"])

        for word in sorted(lexicon):
            for phones, count in lexicon[word].most_common():
                writer.writerow([word, phones, count])

    print_stats(rows, stats, "Таблица без пауз")

    print()
    print("Таблица с паузами")
    print("-----------------")
    print(f"Строк: {len(tokens_sil_df)}")
    print(
        f"Токенов <SIL>: "
        f"{(tokens_sil_df['token'] == SIL).sum()}"
    )

    variants = sum(
        1
        for pronunciations in lexicon.values()
        if len(pronunciations) > 1
    )

    print()
    print("Словарь")
    print("-------")
    print(f"Слов: {len(lexicon)}")
    print(f"Слов с несколькими произношениями: {variants}")

    print()
    print("Сохранено:")
    print(f"  {TOKENS_PATH}")
    print(f"  {TOKENS_SIL_PATH}")
    print(f"  {LEXICON_PATH}")


if __name__ == "__main__":
    main()
