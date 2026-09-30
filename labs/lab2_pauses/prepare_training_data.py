"""Prepare RUSLAN data for pause prediction — lab 2.

The script combines:
1) normalized metadata from lab 1;
2) MFA word alignments from RUSLAN_align_v2.

The result is one row per word:
id | label | label_raw | duration | is_last_word |
is_pause_after | pause_duration | set

Compared with the original skeleton, this version performs a small amount
of conservative cleanup:
- silences shorter than 30 ms are not treated as pauses;
- utterances shorter than 3 words are removed;
- utterances with an implausibly high internal pause density (> 40%) are removed.

We do NOT remove utterances simply because they contain no internal pause:
a perfectly fluent sentence may legitimately have none.
"""

import csv
import os
from pathlib import Path

import pandas as pd
from praatio import textgrid
from tqdm import tqdm


RUSLAN_META = "../../data/metadata_RUSLAN_22200_normalized.csv"
ALIGN_DIR = "../../data/RUSLAN_align_v2"
RESULT_PATH = "data/RUSLAN_pause_metadata.csv"

MIN_PAUSE_DURATION = 0.03   # 30 ms
MIN_WORDS = 3
MAX_PAUSE_DENSITY = 0.40


def read_text_grids(
    ruslan: pd.DataFrame,
    align_root: str,
) -> pd.DataFrame:
    """Read the word tier from MFA TextGrid files."""

    rows = []

    for wave_id in tqdm(ruslan["id"], desc="Читаем TextGrid"):
        path = os.path.join(align_root, f"{wave_id}.TextGrid")

        if not os.path.exists(path):
            continue

        try:
            tg = textgrid.openTextgrid(path, includeEmptyIntervals=True)
            words = tg.getTier("words").entries
        except Exception:
            continue

        for interval in words:
            rows.append(
                {
                    "id": wave_id,
                    "label": interval.label,
                    "duration": float(interval.end - interval.start),
                }
            )

    return pd.DataFrame(rows)


def align_text_and_textgrid(
    tokens: pd.DataFrame,
    text: str,
) -> pd.DataFrame:
    """Attach punctuation-preserving text tokens as label_raw.

    MFA words are lowercase and do not keep punctuation.  This function
    walks through the normalized sentence and attaches punctuation to the
    preceding word.
    """

    if len(tokens) == 0:
        return tokens

    tokens = tokens.copy()
    raw_tokens = []

    text_work = str(text)
    text_lower = text_work.lower()
    previous_word = -1

    for token in tokens["label"].astype(str):
        # Empty TextGrid interval = silence.
        if token == "":
            raw_tokens.append("<SIL>")
            continue

        parts = text_lower.split(token.lower(), maxsplit=1)

        # If MFA token cannot be found in normalized text, alignment is unsafe.
        if len(parts) == 1:
            return tokens

        prefix = parts[0]

        if previous_word == -1:
            raw_tokens.append(
                text_work[: len(prefix) + len(token)].strip()
            )
        else:
            raw_tokens[previous_word] += text_work[: len(prefix)].strip()
            raw_tokens.append(
                text_work[len(prefix): len(prefix) + len(token)]
            )

        consumed = len(prefix) + len(token)
        text_work = text_work[consumed:]
        text_lower = parts[1]
        previous_word = len(raw_tokens) - 1

    if text_work and previous_word >= 0:
        raw_tokens[previous_word] += text_work.strip()

    tokens["label_raw"] = raw_tokens
    return tokens


def add_pause_labels(
    align: pd.DataFrame,
    min_pause_duration: float = MIN_PAUSE_DURATION,
) -> pd.DataFrame:
    """Move silence information onto the preceding word."""

    align = align.copy()

    is_last_word = [False] * len(align)
    pause_after = [False] * len(align)
    pause_duration = [0.0] * len(align)

    last_word = -1

    for idx, row in enumerate(align[["label", "duration"]].itertuples(index=False)):
        label = str(row.label)
        duration = float(row.duration)

        if label == "":
            if last_word >= 0 and duration >= min_pause_duration:
                pause_after[last_word] = True
                pause_duration[last_word] = duration
        else:
            last_word = idx

    if last_word >= 0:
        is_last_word[last_word] = True

    align["is_last_word"] = is_last_word
    align["is_pause_after"] = pause_after
    align["pause_duration"] = pause_duration

    return align


def is_good_utterance(group: pd.DataFrame) -> tuple[bool, str]:
    """Simple segmentation quality filter."""

    words = group[group["label"].astype(str) != ""]

    if len(words) < MIN_WORDS:
        return False, "too_short"

    # The last word is excluded because the final silence belongs to the
    # recording boundary rather than natural within-sentence prosody.
    internal = words[words["is_last_word"] == 0]

    if len(internal) == 0:
        return False, "too_short"

    pause_density = internal["is_pause_after"].mean()

    if pause_density > MAX_PAUSE_DENSITY:
        return False, "too_dense"

    return True, "ok"


def main() -> None:
    Path("data").mkdir(exist_ok=True)

    ruslan = pd.read_csv(
        RUSLAN_META,
        sep="|",
        names=["id", "raw", "nrm"],
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    print(f"Нормализованных utterances из ЛР1: {len(ruslan)}")

    word_df = read_text_grids(ruslan, ALIGN_DIR)

    # MFA and our normalizer may use slightly different Unicode hyphens.
    word_df["label"] = (
        word_df["label"]
        .astype(str)
        .str.replace("‐", "-", regex=False)
        .str.replace("‑", "-", regex=False)
    )

    ruslan["nrm"] = (
        ruslan["nrm"]
        .astype(str)
        .str.replace("‐", "-", regex=False)
        .str.replace("‑", "-", regex=False)
        .str.replace("’", "'", regex=False)
    )

    aligned = []
    failed_alignment = 0

    for row in tqdm(
        ruslan[["id", "nrm"]].itertuples(index=False),
        total=len(ruslan),
        desc="Совмещаем текст и MFA",
    ):
        tokens = word_df[word_df["id"] == row.id].copy()

        if len(tokens) == 0:
            failed_alignment += 1
            continue

        tokens = align_text_and_textgrid(tokens, row.nrm)

        if "label_raw" not in tokens.columns:
            failed_alignment += 1
            continue

        tokens = add_pause_labels(tokens)
        aligned.append(tokens)

    if not aligned:
        raise RuntimeError(
            "Не удалось собрать ни одного utterance. "
            "Проверьте пути к metadata и RUSLAN_align_v2."
        )

    full = pd.concat(aligned, ignore_index=True)

    # Decide which utterances have obviously suspicious segmentation.
    keep_ids = []
    reasons = {"too_short": 0, "too_dense": 0}

    for wave_id, group in full.groupby("id", sort=False):
        ok, reason = is_good_utterance(group)

        if ok:
            keep_ids.append(wave_id)
        else:
            reasons[reason] += 1

    before_utterances = full["id"].nunique()
    before_tokens = len(full)

    pause_df = full[full["id"].isin(keep_ids)].copy()

    # Silence rows are removed; all pause information is already stored
    # on the preceding word.
    pause_df = pause_df[pause_df["label"].astype(str) != ""].copy()
    pause_df = pause_df[pause_df["label_raw"].notna()].copy()
    pause_df.reset_index(drop=True, inplace=True)

    pause_df["is_last_word"] = pause_df["is_last_word"].astype(int)
    pause_df["is_pause_after"] = pause_df["is_pause_after"].astype(int)

    # Fixed 80/20 split from the assignment: id divisible by 5 -> test.
    numeric_id = (
        pause_df["id"]
        .str.split("_", expand=True)[0]
        .astype(int)
    )

    pause_df["set"] = "train"
    pause_df.loc[numeric_id % 5 == 0, "set"] = "test"

    columns = [
        "id",
        "label",
        "label_raw",
        "duration",
        "is_last_word",
        "is_pause_after",
        "pause_duration",
        "set",
    ]

    pause_df[columns].to_csv(
        RESULT_PATH,
        sep="|",
        index=False,
        header=True,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    print()
    print("До фильтрации:")
    print(
        f"  {before_utterances} предложений, "
        f"{before_tokens} интервалов"
    )
    print()
    print("Фильтрация:")
    print(f"  failed_alignment: {failed_alignment}")
    print(f"  too_short (<{MIN_WORDS} слов): {reasons['too_short']}")
    print(
        f"  too_dense (>{MAX_PAUSE_DENSITY:.0%} внутренних пауз): "
        f"{reasons['too_dense']}"
    )
    print()
    print("После фильтрации:")
    print(
        f"  {pause_df['id'].nunique()} предложений, "
        f"{len(pause_df)} слов"
    )

    for split in ["train", "test"]:
        part = pause_df[pause_df["set"] == split]
        print(
            f"  {split}: {len(part)} токенов, "
            f"доля пауз {part['is_pause_after'].mean():.3f}"
        )

    print(f"\nSaved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
