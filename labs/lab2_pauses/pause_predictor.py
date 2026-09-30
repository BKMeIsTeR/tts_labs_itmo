"""Предиктор пауз для лабораторной работы №2.

Здесь я последовательно проверяю четыре подхода из задания:

1. Простые правила по пунктуации.
2. Семантические эмбеддинги RuBERT + лёгкие линейные модели.
3. Морфологические и синтаксические признаки + градиентный бустинг.
4. Последовательная модель на основе BiGRU.

Все четыре подхода сравниваю на одной и той же validation-части.
Test заранее не использую: он нужен только один раз после того, как
финальный подход уже выбран.

Последнее слово предложения не учитываю при обучении и оценке, потому что
пауза после него может относиться просто к границе аудиофайла.

MAE считаю так же, как требуется в задании: только для тех мест, где
реальная пауза была и модель тоже предсказала паузу.

Выбираю не самую сложную модель, а самый простой подход, который на
validation выполняет оба условия:
    F1 >= 0.70
    MAE < 0.120 с

Если ни один подход не проходит оба порога, беру лучший по validation F1.

Обычный запуск сравнения:
    python pause_predictor.py --benchmark

Если нужно переобучить модели заново:
    python pause_predictor.py --benchmark --retrain
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------
# Пути к файлам и общие настройки
# ---------------------------------------------------------------------

DATA_PATH = "data/RUSLAN_pause_metadata.csv"
MODELS_DIR = Path("models")
COMPARISON_PATH = Path("data/approach_comparison.csv")
PREDICTIONS_PATH = Path("data/pause_predictions.csv")
SELECTED_MODEL_PATH = MODELS_DIR / "selected_pause_model.json"

RUBERT_NAME = "cointegrated/rubert-tiny2"

SEED = 42
VAL_FRACTION = 0.10

TARGET_F1 = 0.70
TARGET_MAE = 0.120

DUR_MIN = 0.03
DUR_MAX = 1.50
LOG_EPS = 1e-3

PUNCT_LIST = [",", ".", "—", "–", "-", "?", "!", ";", ":"]
PUNCT_INDEX = {p: i for i, p in enumerate(PUNCT_LIST)}

POS_CATS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB", "X",
]

DEP_CATS = [
    "ROOT", "nsubj", "obj", "iobj", "obl", "advmod", "amod", "nmod",
    "case", "cc", "conj", "mark", "aux", "cop", "xcomp", "ccomp",
    "acl", "advcl", "det", "appos", "parataxis", "punct", "flat",
    "fixed", "orphan", "discourse", "expl", "vocative", "compound",
    "dep", "other",
]

CASE_CATS = ["Nom", "Gen", "Dat", "Acc", "Ins", "Loc", "Voc"]
NUMBER_CATS = ["Sing", "Plur"]
GENDER_CATS = ["Masc", "Fem", "Neut"]
TENSE_CATS = ["Past", "Pres", "Fut"]
VERBFORM_CATS = ["Fin", "Inf", "Part", "Conv"]
PERSON_CATS = ["1", "2", "3"]

CONJ_WORDS = {"и", "а", "но", "или", "либо", "то"}
PART_WORDS = {"же", "ли", "бы", "не", "ни", "вот", "вон", "ведь", "уж"}


# Чем меньше число, тем проще подход объяснить и поддерживать.
COMPLEXITY = {
    "rules": 1,
    "morph_syntax": 2,
    "semantic": 3,
    "seq2seq": 4,
}


# ---------------------------------------------------------------------
# Фиксируем случайность и считаем метрики
# ---------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_train_validation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Делю обучающую часть на fit и validation по целым предложениям.

    Важно делить именно по id предложений, а не перемешивать отдельные слова.
    Официальный test здесь не трогаю - он понадобится только в самом конце.
    """

    # Сначала отделяю официальный test. Его дальше не использую для подбора модели.
    train_df = df[df["set"] == "train"].copy()
    test_df = df[df["set"] == "test"].copy()

    ids = train_df["id"].unique()
    rng = np.random.default_rng(SEED)

    n_val = max(1, int(len(ids) * VAL_FRACTION))
    val_ids = set(rng.choice(ids, size=n_val, replace=False))

    fit_df = train_df[~train_df["id"].isin(val_ids)].copy()
    val_df = train_df[train_df["id"].isin(val_ids)].copy()

    return fit_df, val_df, test_df


def calculate_metrics(
    df: pd.DataFrame,
    print_result: bool = True,
) -> dict[str, float]:
    """Считаю метрики, которые нужны в лабораторной.

    Последние слова предложений к этому моменту уже должны быть исключены.

    MAE считаю только для True Positive:
    пауза есть в разметке и модель тоже предсказала паузу.
    """

    precision = precision_score(
        df["is_pause_after"],
        df["is_pause_hat"],
        zero_division=0,
    )

    recall = recall_score(
        df["is_pause_after"],
        df["is_pause_hat"],
        zero_division=0,
    )

    f1 = f1_score(
        df["is_pause_after"],
        df["is_pause_hat"],
        zero_division=0,
    )

    tp = df[
        (df["is_pause_after"] == 1)
        & (df["is_pause_hat"] == 1)
    ]

    if len(tp):
        mae = mean_absolute_error(
            tp["pause_duration"],
            tp["pause_duration_hat"],
        )
    else:
        mae = float("nan")

    result = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mae": float(mae),
        "tp": int(len(tp)),
    }

    if print_result:
        print(
            f"PRC: {precision:.4f}, "
            f"REC: {recall:.4f}, "
            f"F1: {f1:.4f}; "
            f"MAE: {mae:.4f}; "
            f"TP: {len(tp)}"
        )

    return result


def tune_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """Подбираю порог классификации только по F1 на validation."""

    best_threshold = 0.5
    best_f1 = -1.0

    # Проверяю несколько порогов от 0.20 до 0.80 и оставляю тот,
    # который даёт лучший F1 на validation.
    for threshold in np.arange(0.20, 0.81, 0.02):
        pred = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, pred, zero_division=0)

        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def run_predictor(
    predictor,
    part: pd.DataFrame,
    name: str = "predict",
    verbose: bool = True,
) -> pd.DataFrame:
    """Запускаю любой из предикторов по одному предложению за раз."""

    part = part.copy()

    pause_hat = []
    duration_hat = []

    groups = part.groupby("id", sort=False)

    if verbose:
        groups = tqdm(groups, desc=name)

    for _, group in groups:
        is_pause, durations = predictor.predict(
            group["label_raw"].astype(str).values
        )

        pause_hat.extend(is_pause.tolist())
        duration_hat.extend(durations.tolist())

    part["is_pause_hat"] = pause_hat
    part["pause_duration_hat"] = duration_hat

    return part


# ---------------------------------------------------------------------
# Вспомогательные функции для текста и пунктуации
# ---------------------------------------------------------------------

def split_word_punct(
    label_raw: str,
) -> tuple[str, np.ndarray, str]:
    """Отделяю слово от пунктуации и возвращаю признаки знаков препинания."""

    text = str(label_raw).strip()
    text = re.sub(r'[«»"“”„]', "", text)
    text = re.sub(r"\s+", " ", text).strip()

    match = re.search(r"([.,!?;:—–\-]+)\s*$", text)

    punct = np.zeros(len(PUNCT_LIST), dtype=np.float32)

    if match:
        punct_string = match.group(1)

        for char in punct_string:
            if char in PUNCT_INDEX:
                punct[PUNCT_INDEX[char]] = 1.0

        word = text[: match.start()].strip()
        category = punct_string
    else:
        word = text
        category = "(нет)"

    return word.lower(), punct, category


def basic_features_for_tokens(
    tokens: list[str] | np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Собираю небольшой общий набор числовых признаков для слова."""

    n = len(tokens)

    words = []
    puncts = []

    for token in tokens:
        word, punct, _ = split_word_punct(token)
        words.append(word)
        puncts.append(punct)

    # Здесь собираю простые признаки: знак препинания, позицию и длину слова,
    # союз/частицу, пунктуацию у соседей и длину предложения.
    extra_dim = 7
    features = np.zeros(
        (n, len(PUNCT_LIST) + extra_dim),
        dtype=np.float32,
    )

    for i, word in enumerate(words):
        features[i, :len(PUNCT_LIST)] = puncts[i]

        base = len(PUNCT_LIST)

        features[i, base + 0] = i / max(n - 1, 1)
        features[i, base + 1] = min(len(word), 20) / 20.0
        features[i, base + 2] = float(word in CONJ_WORDS)
        features[i, base + 3] = float(word in PART_WORDS)
        features[i, base + 4] = float(i > 0 and puncts[i - 1].any())
        features[i, base + 5] = float(i + 1 < n and puncts[i + 1].any())
        features[i, base + 6] = min(n, 50) / 50.0

    return words, features


def flatten_basic_features(
    df: pd.DataFrame,
) -> tuple[list[str], np.ndarray]:
    """Собираю слова и признаки в том же порядке, что и строки DataFrame."""

    all_words = []
    all_features = []

    for _, group in df.groupby("id", sort=False):
        words, features = basic_features_for_tokens(
            group["label_raw"].astype(str).values
        )

        all_words.extend(words)
        all_features.append(features)

    return all_words, np.vstack(all_features)


# ---------------------------------------------------------------------
# 1. Подход на правилах
# ---------------------------------------------------------------------

class RulePausePredictor:
    """Самый простой baseline: пунктуация + статистика пауз из train."""

    name = "rules"

    def __init__(self):
        self.pause_probability: dict[str, float] = {}
        self.pause_median: dict[str, float] = {}
        self.global_pause_probability = 0.0
        self.global_pause_median = 0.12
        self.threshold = 0.5

    @staticmethod
    def _category(token: str) -> str:
        return split_word_punct(token)[2]

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        train = train_df[train_df["is_last_word"] == 0].copy()
        val = val_df[val_df["is_last_word"] == 0].copy()

        train["punct_category"] = train["label_raw"].apply(self._category)
        val["punct_category"] = val["label_raw"].apply(self._category)

        grouped = train.groupby("punct_category")

        self.pause_probability = (
            grouped["is_pause_after"]
            .mean()
            .to_dict()
        )

        self.pause_median = (
            train[train["is_pause_after"] == 1]
            .groupby("punct_category")["pause_duration"]
            .median()
            .to_dict()
        )

        self.global_pause_probability = float(
            train["is_pause_after"].mean()
        )

        self.global_pause_median = float(
            train.loc[
                train["is_pause_after"] == 1,
                "pause_duration",
            ].median()
        )

        val_prob = (
            val["punct_category"]
            .map(self.pause_probability)
            .fillna(self.global_pause_probability)
            .to_numpy(float)
        )

        self.threshold, _ = tune_threshold(
            val["is_pause_after"].to_numpy(int),
            val_prob,
        )

    def predict(
        self,
        tokens: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        is_pause = np.zeros(len(tokens), dtype=int)
        durations = np.zeros(len(tokens), dtype=float)

        for i, token in enumerate(tokens):
            category = self._category(str(token))

            probability = self.pause_probability.get(
                category,
                self.global_pause_probability,
            )

            if probability >= self.threshold:
                is_pause[i] = 1

                durations[i] = self.pause_median.get(
                    category,
                    self.global_pause_median,
                )

        durations = np.clip(
            durations,
            0.0,
            DUR_MAX,
        )

        return is_pause, durations

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "pause_probability": self.pause_probability,
                "pause_median": self.pause_median,
                "global_pause_probability": self.global_pause_probability,
                "global_pause_median": self.global_pause_median,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        state = joblib.load(path)
        model = cls()

        for key, value in state.items():
            setattr(model, key, value)

        return model


# ---------------------------------------------------------------------
# Замороженные эмбеддинги RuBERT
# ---------------------------------------------------------------------

class FrozenRuBERTEmbedder:
    """Получаю один вектор RuBERT для каждого слова.

    Слово может разбиться токенизатором на несколько subword-частей.
    Я не беру только первую часть, а усредняю представления всех частей слова.

    RuBERT здесь не дообучаю. Уже посчитанные векторы сохраняю в кэше,
    чтобы не считать одно и то же слово много раз.
    """

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(RUBERT_NAME)
        self.model = AutoModel.from_pretrained(RUBERT_NAME).to(self.device)
        self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        self.dim = int(self.model.config.hidden_size)
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _encode_missing(
        self,
        words: list[str],
        batch_size: int = 256,
    ) -> None:
        missing = list(
            dict.fromkeys(
                word if word else "[UNK]"
                for word in words
                if (word if word else "[UNK]") not in self.cache
            )
        )

        for start in tqdm(
            range(0, len(missing), batch_size),
            desc="RuBERT embeddings",
            leave=False,
        ):
            batch_words = missing[start:start + batch_size]

            encoded = self.tokenizer(
                batch_words,
                padding=True,
                truncation=True,
                max_length=24,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].clone().bool()

            # Служебный токен [CLS] в представление слова не включаю.
            mask[:, 0] = False

            # Служебный [SEP] в конце слова тоже убираю.
            last_index = (
                encoded["attention_mask"].sum(dim=1) - 1
            )

            mask[
                torch.arange(len(batch_words), device=self.device),
                last_index,
            ] = False

            weights = mask.unsqueeze(-1).float()
            denom = weights.sum(dim=1).clamp(min=1.0)

            pooled = (
                (hidden * weights).sum(dim=1)
                / denom
            )

            pooled = pooled.cpu().numpy().astype(np.float32)

            for word, vector in zip(batch_words, pooled):
                self.cache[word] = vector

    def encode_words(
        self,
        words: list[str],
    ) -> np.ndarray:
        keys = [
            word if word else "[UNK]"
            for word in words
        ]

        self._encode_missing(keys)

        return np.vstack(
            [self.cache[key] for key in keys]
        ).astype(np.float32)


_GLOBAL_EMBEDDER: FrozenRuBERTEmbedder | None = None


def get_embedder() -> FrozenRuBERTEmbedder:
    global _GLOBAL_EMBEDDER

    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = FrozenRuBERTEmbedder()

    return _GLOBAL_EMBEDDER


# ---------------------------------------------------------------------
# 2. Подход с семантическими эмбеддингами
# ---------------------------------------------------------------------

class SemanticPausePredictor:
    """Семантический подход: замороженный RuBERT + простые линейные модели."""

    name = "semantic"

    def __init__(self):
        self.classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            max_iter=1500,
            tol=1e-4,
            class_weight="balanced",
            random_state=SEED,
        )

        self.regressor = Ridge(alpha=10.0)
        self.threshold = 0.5
        self.embedding_dim = None

    def _matrix(
        self,
        df: pd.DataFrame,
    ) -> np.ndarray:
        words, basic = flatten_basic_features(df)

        embeddings = get_embedder().encode_words(words)
        self.embedding_dim = embeddings.shape[1]

        return np.hstack(
            [embeddings, basic]
        ).astype(np.float32)

    def _sentence_matrix(
        self,
        tokens: list[str] | np.ndarray,
    ) -> np.ndarray:
        words, basic = basic_features_for_tokens(tokens)
        embeddings = get_embedder().encode_words(words)

        return np.hstack(
            [embeddings, basic]
        ).astype(np.float32)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        train_mask = train_df["is_last_word"].to_numpy() == 0
        val_mask = val_df["is_last_word"].to_numpy() == 0

        x_train_all = self._matrix(train_df)
        x_val_all = self._matrix(val_df)

        x_train = x_train_all[train_mask]
        y_train = train_df.loc[
            train_df["is_last_word"] == 0,
            "is_pause_after",
        ].to_numpy(int)

        x_val = x_val_all[val_mask]
        y_val = val_df.loc[
            val_df["is_last_word"] == 0,
            "is_pause_after",
        ].to_numpy(int)

        # Классификатор отвечает только за вопрос "пауза / нет паузы".
        self.classifier.fit(x_train, y_train)

        val_prob = self.classifier.predict_proba(x_val)[:, 1]

        self.threshold, _ = tune_threshold(
            y_val,
            val_prob,
        )

        pause_train = (
            (train_df["is_last_word"].to_numpy() == 0)
            & (train_df["is_pause_after"].to_numpy() == 1)
        )

        y_duration = np.log(
            train_df.loc[
                pause_train,
                "pause_duration",
            ].to_numpy(float)
            + LOG_EPS
        )

        # Регрессию длительности обучаю только на реальных паузах.
        self.regressor.fit(
            x_train_all[pause_train],
            y_duration,
        )

    def predict(
        self,
        tokens: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self._sentence_matrix(tokens)

        probability = self.classifier.predict_proba(x)[:, 1]
        is_pause = (probability >= self.threshold).astype(int)

        log_duration = self.regressor.predict(x)
        duration = np.exp(log_duration) - LOG_EPS

        duration = np.where(
            is_pause == 1,
            np.clip(duration, DUR_MIN, DUR_MAX),
            0.0,
        )

        return is_pause, duration

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "classifier": self.classifier,
                "regressor": self.regressor,
                "threshold": self.threshold,
                "embedding_dim": self.embedding_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        state = joblib.load(path)
        model = cls()
        model.classifier = state["classifier"]
        model.regressor = state["regressor"]
        model.threshold = state["threshold"]
        model.embedding_dim = state["embedding_dim"]
        return model


# ---------------------------------------------------------------------
# 3. Морфологические и синтаксические признаки
# ---------------------------------------------------------------------

_SPACY = None
_MORPH_CACHE: dict[tuple[str, ...], np.ndarray] = {}


def get_spacy():
    """Загружаю spaCy с POS, морфологией и синтаксическим парсером."""

    global _SPACY

    if _SPACY is None:
        import spacy

        # Парсер и морфологию оставляю, а NER здесь не нужен.
        _SPACY = spacy.load(
            "ru_core_news_sm",
            disable=["ner"],
        )

    return _SPACY


def one_hot(
    value: str,
    categories: list[str],
) -> np.ndarray:
    result = np.zeros(len(categories), dtype=np.float32)

    if value in categories:
        result[categories.index(value)] = 1.0

    return result


def first_morph_value(token, key: str) -> str:
    values = token.morph.get(key)
    return values[0] if values else ""


def morph_syntax_features_for_tokens(
    tokens: list[str] | np.ndarray,
) -> np.ndarray:
    """Получаю морфологические и синтаксические признаки для каждого слова.

    Токены заранее фиксирую по словам MFA, чтобы признаки не сдвинулись
    относительно правильных меток пауз.

    Результат сохраняю в кэше, потому что эти же признаки потом использует
    и последовательная модель.
    """

    key = tuple(str(token) for token in tokens)

    if key in _MORPH_CACHE:
        return _MORPH_CACHE[key]

    from spacy.tokens import Doc

    nlp = get_spacy()

    words, basic = basic_features_for_tokens(tokens)

    safe_words = [
        word if word else "пусто"
        for word in words
    ]

    # Важно сохранить соответствие: один токен spaCy = одно слово из MFA.
    doc = Doc(
        nlp.vocab,
        words=safe_words,
    )

    for pipe_name, pipe in nlp.pipeline:
        # Токены уже заданы вручную через Doc(...), поэтому заново делить текст не нужно.
        doc = pipe(doc)

    rows = []

    for i, token in enumerate(doc):
        row = [*basic[i]]

        cur_pos = token.pos_ if token.pos_ in POS_CATS else "X"
        prev_pos = (
            doc[i - 1].pos_
            if i > 0 and doc[i - 1].pos_ in POS_CATS
            else "X"
        )
        next_pos = (
            doc[i + 1].pos_
            if i + 1 < len(doc) and doc[i + 1].pos_ in POS_CATS
            else "X"
        )

        row.extend(one_hot(cur_pos, POS_CATS))
        row.extend(one_hot(prev_pos, POS_CATS))
        row.extend(one_hot(next_pos, POS_CATS))

        dep = token.dep_ if token.dep_ in DEP_CATS else "other"
        row.extend(one_hot(dep, DEP_CATS))

        # Добавляю направление синтаксической связи и число зависимых слов.
        row.extend([
            float(token.head.i < token.i),
            float(token.head.i > token.i),
            min(len(list(token.children)), 5) / 5.0,
        ])

        row.extend(
            one_hot(
                first_morph_value(token, "Case"),
                CASE_CATS,
            )
        )

        row.extend(
            one_hot(
                first_morph_value(token, "Number"),
                NUMBER_CATS,
            )
        )

        row.extend(
            one_hot(
                first_morph_value(token, "Gender"),
                GENDER_CATS,
            )
        )

        row.extend(
            one_hot(
                first_morph_value(token, "Tense"),
                TENSE_CATS,
            )
        )

        row.extend(
            one_hot(
                first_morph_value(token, "VerbForm"),
                VERBFORM_CATS,
            )
        )

        row.extend(
            one_hot(
                first_morph_value(token, "Person"),
                PERSON_CATS,
            )
        )

        rows.append(row)

    result = np.asarray(rows, dtype=np.float32)
    _MORPH_CACHE[key] = result
    return result


def morph_syntax_matrix(
    df: pd.DataFrame,
) -> np.ndarray:
    matrices = []

    groups = df.groupby("id", sort=False)

    for _, group in tqdm(
        groups,
        total=df["id"].nunique(),
        desc="spaCy features",
        leave=False,
    ):
        matrices.append(
            morph_syntax_features_for_tokens(
                group["label_raw"].astype(str).values
            )
        )

    return np.vstack(matrices)


class MorphSyntaxPausePredictor:
    """Морфология, синтаксис и пунктуация + градиентный бустинг."""

    name = "morph_syntax"

    def __init__(self):
        self.classifier = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=160,
            max_leaf_nodes=31,
            l2_regularization=0.2,
            random_state=SEED,
        )

        self.regressor = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.06,
            max_iter=140,
            max_leaf_nodes=25,
            l2_regularization=0.2,
            random_state=SEED,
        )

        self.threshold = 0.5

    @staticmethod
    def _balanced_weights(y: np.ndarray) -> np.ndarray:
        n = len(y)
        positive = max(int((y == 1).sum()), 1)
        negative = max(int((y == 0).sum()), 1)

        weights = np.ones(n, dtype=np.float32)
        weights[y == 1] = n / (2.0 * positive)
        weights[y == 0] = n / (2.0 * negative)

        return weights

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        x_train_all = morph_syntax_matrix(train_df)
        x_val_all = morph_syntax_matrix(val_df)

        train_mask = train_df["is_last_word"].to_numpy() == 0
        val_mask = val_df["is_last_word"].to_numpy() == 0

        x_train = x_train_all[train_mask]

        y_train = train_df.loc[
            train_df["is_last_word"] == 0,
            "is_pause_after",
        ].to_numpy(int)

        x_val = x_val_all[val_mask]

        y_val = val_df.loc[
            val_df["is_last_word"] == 0,
            "is_pause_after",
        ].to_numpy(int)

        # Пауз заметно меньше, чем обычных слов, поэтому использую веса классов.
        self.classifier.fit(
            x_train,
            y_train,
            sample_weight=self._balanced_weights(y_train),
        )

        val_prob = self.classifier.predict_proba(x_val)[:, 1]

        self.threshold, _ = tune_threshold(
            y_val,
            val_prob,
        )

        pause_train = (
            (train_df["is_last_word"].to_numpy() == 0)
            & (train_df["is_pause_after"].to_numpy() == 1)
        )

        # Второй бустинг отдельно учится предсказывать длительность только пауз.
        self.regressor.fit(
            x_train_all[pause_train],
            train_df.loc[
                pause_train,
                "pause_duration",
            ].to_numpy(float),
        )

    def predict(
        self,
        tokens: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = morph_syntax_features_for_tokens(tokens)

        probability = self.classifier.predict_proba(x)[:, 1]
        is_pause = (probability >= self.threshold).astype(int)

        duration = self.regressor.predict(x)

        duration = np.where(
            is_pause == 1,
            np.clip(duration, DUR_MIN, DUR_MAX),
            0.0,
        )

        return is_pause, duration

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {
                "classifier": self.classifier,
                "regressor": self.regressor,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        state = joblib.load(path)
        model = cls()
        model.classifier = state["classifier"]
        model.regressor = state["regressor"]
        model.threshold = state["threshold"]
        return model


# ---------------------------------------------------------------------
# 4. Последовательная модель (Sequence2Sequence / sequence labelling)
# ---------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Одно предложение на входе -> последовательность меток такой же длины."""

    def __init__(
        self,
        df: pd.DataFrame,
    ):
        self.samples = []

        for _, group in df.groupby("id", sort=False):
            tokens = group["label_raw"].astype(str).values

            words, _ = basic_features_for_tokens(tokens)
            embeddings = get_embedder().encode_words(words)
            linguistic = morph_syntax_features_for_tokens(tokens)

            x = np.hstack(
                [embeddings, linguistic]
            ).astype(np.float32)

            y = group["is_pause_after"].to_numpy(np.int64)

            duration = group["pause_duration"].to_numpy(np.float32)
            duration_log = np.log(
                duration + LOG_EPS
            ).astype(np.float32)

            is_last = group["is_last_word"].to_numpy(bool)

            self.samples.append(
                (
                    x,
                    y,
                    duration_log,
                    duration,
                    is_last,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_sequences(batch):
    x_b, y_b, logdur_b, dur_b, last_b = zip(*batch)

    lengths = np.asarray(
        [len(x) for x in x_b],
        dtype=np.int64,
    )

    max_len = int(lengths.max())
    batch_size = len(batch)
    input_dim = x_b[0].shape[1]

    x = torch.zeros(
        batch_size,
        max_len,
        input_dim,
        dtype=torch.float32,
    )

    y = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.long,
    )

    logdur = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.float32,
    )

    duration = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.float32,
    )

    valid = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.bool,
    )

    is_last = torch.zeros(
        batch_size,
        max_len,
        dtype=torch.bool,
    )

    for i in range(batch_size):
        n = len(x_b[i])

        x[i, :n] = torch.tensor(x_b[i])
        y[i, :n] = torch.tensor(y_b[i])
        logdur[i, :n] = torch.tensor(logdur_b[i])
        duration[i, :n] = torch.tensor(dur_b[i])
        valid[i, :n] = True
        is_last[i, :n] = torch.tensor(last_b[i])

    return (
        x,
        y,
        logdur,
        duration,
        valid,
        is_last,
        torch.tensor(lengths, dtype=torch.long),
    )


class _BiGRUNet(nn.Module):
    """Небольшая последовательная модель. Сам RuBERT здесь остаётся замороженным."""

    def __init__(
        self,
        input_dim: int,
        hidden: int = 64,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(hidden * 2, 2)
        self.regressor = nn.Linear(hidden * 2, 1)

    def forward(
        self,
        x,
        lengths,
    ):
        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.gru(packed)

        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.shape[1],
        )

        out = self.dropout(out)

        logits = self.classifier(out)
        log_duration = self.regressor(out).squeeze(-1)

        return logits, log_duration


class Seq2SeqPausePredictor:
    """Упрощённый последовательный подход.

    Для каждого слова на входе модель выдаёт:
    - есть ли после него пауза;
    - какой должна быть её длительность.

    RuBERT использую только для готовых векторов слов и не дообучаю.
    Обучаются только небольшая BiGRU и две выходные головы.
    """

    name = "seq2seq"

    def __init__(self):
        set_seed()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model: _BiGRUNet | None = None
        self.input_dim: int | None = None
        self.threshold = 0.5

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 4,
        batch_size: int = 32,
    ) -> None:
        train_dataset = SequenceDataset(train_df)
        val_dataset = SequenceDataset(val_df)

        self.input_dim = train_dataset[0][0].shape[1]

        self.model = _BiGRUNet(
            input_dim=self.input_dim,
            hidden=64,
        ).to(self.device)

        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_sequences,
        )

        internal_train = train_df[
            train_df["is_last_word"] == 0
        ]

        positive = max(
            int(internal_train["is_pause_after"].sum()),
            1,
        )

        negative = max(
            len(internal_train) - positive,
            1,
        )

        # Положительный класс встречается реже, поэтому увеличиваю его вес в loss.
        class_weights = torch.tensor(
            [1.0, negative / positive],
            dtype=torch.float32,
            device=self.device,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        best_state = None
        best_f1 = -1.0
        best_mae = float("inf")
        best_threshold = 0.5

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in loader:
                (
                    x,
                    y,
                    logdur,
                    _,
                    valid,
                    is_last,
                    lengths,
                ) = batch

                x = x.to(self.device)
                y = y.to(self.device)
                logdur = logdur.to(self.device)
                valid = valid.to(self.device)
                is_last = is_last.to(self.device)
                lengths = lengths.to(self.device)

                eval_mask = valid & (~is_last)

                optimizer.zero_grad()

                logits, pred_logdur = self.model(
                    x,
                    lengths,
                )

                # Первая часть loss отвечает за место паузы.
                classification_loss = F.cross_entropy(
                    logits[eval_mask],
                    y[eval_mask],
                    weight=class_weights,
                )

                pause_mask = eval_mask & (y == 1)

                # Вторую часть loss считаю только там, где пауза реально есть.
                if pause_mask.any():
                    duration_loss = F.smooth_l1_loss(
                        pred_logdur[pause_mask],
                        logdur[pause_mask],
                    )
                else:
                    duration_loss = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                loss = classification_loss + 0.35 * duration_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    5.0,
                )
                optimizer.step()

                epoch_loss += float(loss.item())

            val_prob, val_duration, val_true = self._predict_dataframe_raw(
                val_df
            )

            threshold, val_f1 = tune_threshold(
                val_true["is_pause_after"].to_numpy(int),
                val_prob,
            )

            val_pred = (val_prob >= threshold).astype(int)

            temp = val_true.copy()
            temp["is_pause_hat"] = val_pred
            temp["pause_duration_hat"] = np.where(
                val_pred == 1,
                val_duration,
                0.0,
            )

            metrics = calculate_metrics(
                temp,
                print_result=False,
            )

            print(
                f"seq2seq epoch {epoch + 1}/{epochs}: "
                f"loss={epoch_loss / max(len(loader), 1):.4f}, "
                f"val_f1={metrics['f1']:.4f}, "
                f"val_mae={metrics['mae']:.4f}"
            )

            better = (
                metrics["f1"] > best_f1 + 1e-4
                or (
                    abs(metrics["f1"] - best_f1) <= 1e-4
                    and metrics["mae"] < best_mae
                )
            )

            if better:
                best_f1 = metrics["f1"]
                best_mae = metrics["mae"]
                best_threshold = threshold

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.threshold = float(best_threshold)

    @torch.no_grad()
    def _predict_sequence_raw(
        self,
        tokens: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Sequence model is not fitted.")

        words, _ = basic_features_for_tokens(tokens)
        embeddings = get_embedder().encode_words(words)
        linguistic = morph_syntax_features_for_tokens(tokens)

        x = np.hstack(
            [embeddings, linguistic]
        ).astype(np.float32)

        x_t = torch.tensor(
            x[None, :, :],
            dtype=torch.float32,
            device=self.device,
        )

        lengths = torch.tensor(
            [len(tokens)],
            dtype=torch.long,
            device=self.device,
        )

        self.model.eval()

        logits, log_duration = self.model(
            x_t,
            lengths,
        )

        probability = (
            F.softmax(logits, dim=-1)[0, :, 1]
            .cpu()
            .numpy()
        )

        duration = (
            np.exp(log_duration[0].cpu().numpy())
            - LOG_EPS
        )

        duration = np.clip(
            duration,
            DUR_MIN,
            DUR_MAX,
        )

        return probability, duration

    def _predict_dataframe_raw(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        probabilities = []
        durations = []
        rows = []

        for _, group in df.groupby("id", sort=False):
            prob, dur = self._predict_sequence_raw(
                group["label_raw"].astype(str).values
            )

            keep = group["is_last_word"].to_numpy() == 0

            probabilities.extend(prob[keep].tolist())
            durations.extend(dur[keep].tolist())
            rows.append(
                group.loc[
                    group["is_last_word"] == 0,
                    [
                        "is_pause_after",
                        "pause_duration",
                    ],
                ]
            )

        return (
            np.asarray(probabilities),
            np.asarray(durations),
            pd.concat(rows, ignore_index=True),
        )

    def predict(
        self,
        tokens: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        probability, duration = self._predict_sequence_raw(tokens)

        is_pause = (
            probability >= self.threshold
        ).astype(int)

        duration = np.where(
            is_pause == 1,
            duration,
            0.0,
        )

        return is_pause, duration

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Nothing to save.")

        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_dim": self.input_dim,
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path):
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        model = cls()
        model.input_dim = int(state["input_dim"])
        model.threshold = float(state["threshold"])

        model.model = _BiGRUNet(
            model.input_dim,
            hidden=64,
        ).to(model.device)

        model.model.load_state_dict(
            state["state_dict"]
        )

        model.model.eval()

        return model


# ---------------------------------------------------------------------
# Сохранение моделей и общий класс-обёртка
# ---------------------------------------------------------------------

MODEL_FILES = {
    "rules": MODELS_DIR / "pause_rules.joblib",
    "morph_syntax": MODELS_DIR / "pause_morph_syntax.joblib",
    "semantic": MODELS_DIR / "pause_semantic.joblib",
    "seq2seq": MODELS_DIR / "pause_seq2seq.pt",
}


def save_predictor(
    name: str,
    predictor,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save(MODEL_FILES[name])


def load_predictor(
    name: str,
):
    if name == "rules":
        return RulePausePredictor.load(MODEL_FILES[name])

    if name == "morph_syntax":
        return MorphSyntaxPausePredictor.load(MODEL_FILES[name])

    if name == "semantic":
        return SemanticPausePredictor.load(MODEL_FILES[name])

    if name == "seq2seq":
        return Seq2SeqPausePredictor.load(MODEL_FILES[name])

    raise ValueError(f"Unknown approach: {name}")


class PausePredictor:
    """Общий класс, который потом можно использовать в TTS-конвейере.

    Он читает результат --benchmark и загружает уже выбранный подход.
    """

    def __init__(self):
        if not SELECTED_MODEL_PATH.exists():
            raise RuntimeError(
                "No selected model yet. "
                "Run: python pause_predictor.py --benchmark"
            )

        config = json.loads(
            SELECTED_MODEL_PATH.read_text(encoding="utf-8")
        )

        self.approach = config["approach"]
        self.predictor = load_predictor(self.approach)

    def predict(
        self,
        tokens: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.predictor.predict(tokens)

    def predict_durations(
        self,
        tokens: list[str] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        is_pause, durations = self.predict(tokens)

        result_tokens = []
        result_durations = []

        for token, pause, duration in zip(
            tokens,
            is_pause,
            durations,
        ):
            result_tokens.append(token)
            result_durations.append(-1.0)

            if pause:
                result_tokens.append("<SIL>")
                result_durations.append(float(duration))

        return (
            np.asarray(result_tokens),
            np.asarray(result_durations, dtype=np.float32),
        )


# ---------------------------------------------------------------------
# Сравнение четырёх подходов
# ---------------------------------------------------------------------

def create_approach(name: str):
    if name == "rules":
        return RulePausePredictor()

    if name == "morph_syntax":
        return MorphSyntaxPausePredictor()

    if name == "semantic":
        return SemanticPausePredictor()

    if name == "seq2seq":
        return Seq2SeqPausePredictor()

    raise ValueError(name)


def validation_row(
    name: str,
    metrics: dict[str, float],
    train_seconds: float,
) -> dict[str, Any]:
    passes = (
        metrics["f1"] >= TARGET_F1
        and metrics["mae"] < TARGET_MAE
    )

    return {
        "approach": name,
        "complexity": COMPLEXITY[name],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
        "val_f1": metrics["f1"],
        "val_mae": metrics["mae"],
        "train_seconds": float(train_seconds),
        "passes_validation": bool(passes),
        "selected": False,
        "test_precision": np.nan,
        "test_recall": np.nan,
        "test_f1": np.nan,
        "test_mae": np.nan,
        "passes_test": np.nan,
    }


def select_approach(
    comparison: pd.DataFrame,
) -> str:
    """Выбираю самый простой подход, который прошёл оба порога на validation."""

    passed = comparison[
        comparison["passes_validation"] == True
    ].copy()

    if len(passed):
        # Если пороги прошли несколько моделей, не усложняю решение без необходимости.
        # Сначала выбираю самый простой подход. Метрики использую только для разрешения равенства.
        passed = passed.sort_values(
            ["complexity", "val_f1", "val_mae"],
            ascending=[True, False, True],
        )

        return str(passed.iloc[0]["approach"])

    # Если ни один подход не прошёл оба порога, беру лучший по validation F1,
    # а при равенстве смотрю на MAE и сложность.
    fallback = comparison.sort_values(
        ["val_f1", "val_mae", "complexity"],
        ascending=[False, True, True],
    )

    return str(fallback.iloc[0]["approach"])


def benchmark(
    force_retrain: bool = False,
) -> None:
    set_seed()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        sep="|",
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    fit_df, val_df, test_df = split_train_validation(df)

    print("Data:")
    print(
        f"  fit utterances: {fit_df['id'].nunique()}, "
        f"tokens: {len(fit_df)}"
    )
    print(
        f"  val utterances: {val_df['id'].nunique()}, "
        f"tokens: {len(val_df)}"
    )
    print(
        f"  test utterances: {test_df['id'].nunique()}, "
        f"tokens: {len(test_df)}"
    )

    print()
    print(
        "All 4 approaches are compared on validation. "
        "Test remains untouched until selection."
    )

    rows = []

    # Порядок такой же, как в задании и в отчёте.
    approach_order = [
        "rules",
        "semantic",
        "morph_syntax",
        "seq2seq",
    ]

    for name in approach_order:
        print()
        print("=" * 72)
        print(f"Approach: {name}")
        print("=" * 72)

        path = MODEL_FILES[name]

        if path.exists() and not force_retrain:
            predictor = load_predictor(name)
            train_seconds = 0.0
            print(f"Loaded: {path}")
        else:
            predictor = create_approach(name)

            start = time.perf_counter()
            predictor.fit(fit_df, val_df)
            train_seconds = time.perf_counter() - start

            save_predictor(name, predictor)

        val_out = run_predictor(
            predictor,
            val_df,
            name=f"val/{name}",
        )

        val_eval = val_out[
            val_out["is_last_word"] == 0
        ]

        metrics = calculate_metrics(val_eval)

        row = validation_row(
            name,
            metrics,
            train_seconds,
        )

        rows.append(row)

    comparison = pd.DataFrame(rows)

    # Явно задаю булев тип, потому что новые версии pandas
    # не разрешают записывать True в столбец float, созданный из np.nan.
    comparison["selected"] = comparison["selected"].astype("boolean")
    comparison["passes_validation"] = comparison["passes_validation"].astype("boolean")
    comparison["passes_test"] = pd.Series(
        pd.NA,
        index=comparison.index,
        dtype="boolean",
    )

    selected = select_approach(comparison)

    comparison.loc[
        comparison["approach"] == selected,
        "selected",
    ] = True

    print()
    print("=" * 72)
    print(f"Selected approach: {selected}")
    print("=" * 72)

    # Test использую только здесь - уже после выбора подхода по validation.
    final_predictor = load_predictor(selected)

    test_out = run_predictor(
        final_predictor,
        test_df,
        name=f"test/{selected}",
    )

    test_eval = test_out[
        test_out["is_last_word"] == 0
    ]

    test_metrics = calculate_metrics(test_eval)

    selected_mask = comparison["approach"] == selected

    comparison.loc[
        selected_mask,
        "test_precision",
    ] = test_metrics["precision"]

    comparison.loc[
        selected_mask,
        "test_recall",
    ] = test_metrics["recall"]

    comparison.loc[
        selected_mask,
        "test_f1",
    ] = test_metrics["f1"]

    comparison.loc[
        selected_mask,
        "test_mae",
    ] = test_metrics["mae"]

    comparison.loc[
        selected_mask,
        "passes_test",
    ] = (
        test_metrics["f1"] >= TARGET_F1
        and test_metrics["mae"] < TARGET_MAE
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    test_out.to_csv(
        PREDICTIONS_PATH,
        sep="|",
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
    )

    SELECTED_MODEL_PATH.write_text(
        json.dumps(
            {
                "approach": selected,
                "targets": {
                    "f1": TARGET_F1,
                    "mae": TARGET_MAE,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Validation comparison:")
    print(
        comparison[
            [
                "approach",
                "val_precision",
                "val_recall",
                "val_f1",
                "val_mae",
                "passes_validation",
                "selected",
            ]
        ].to_string(index=False)
    )

    print()
    print("Final test:")
    print(f"  approach: {selected}")
    print(
        f"  F1: {test_metrics['f1']:.4f} "
        f"(target >= {TARGET_F1:.2f})"
    )
    print(
        f"  MAE: {test_metrics['mae']:.4f} s "
        f"(target < {TARGET_MAE:.3f})"
    )

    print()
    print(f"Saved comparison to: {COMPARISON_PATH}")
    print(f"Saved test predictions to: {PREDICTIONS_PATH}")
    print(f"Saved selected model config to: {SELECTED_MODEL_PATH}")


# ---------------------------------------------------------------------
# Запуск из командной строки
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Train/compare all four approaches and select the final one.",
    )

    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models even if saved files already exist.",
    )

    args = parser.parse_args()

    if args.benchmark:
        benchmark(force_retrain=args.retrain)
    else:
        # Без --benchmark можно быстро проверить уже выбранную и сохранённую модель.
        if not SELECTED_MODEL_PATH.exists():
            print(
                "No selected model yet. Run:\n"
                "  python pause_predictor.py --benchmark"
            )
        else:
            df = pd.read_csv(
                DATA_PATH,
                sep="|",
                encoding="utf-8",
                quoting=csv.QUOTE_NONE,
            )

            _, _, test_df = split_train_validation(df)

            predictor = PausePredictor()

            out = run_predictor(
                predictor,
                test_df,
                name=f"test/{predictor.approach}",
            )

            evaluated = out[
                out["is_last_word"] == 0
            ]

            print(f"Approach: {predictor.approach}")
            calculate_metrics(evaluated)
