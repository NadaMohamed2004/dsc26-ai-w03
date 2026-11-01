"""Tawseel: an honest baseline for failed COD deliveries."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "failed"

# Known when the parcel is created. Excluded on purpose:
#   parcel_id (row identifier), courier_notes + delivered_at (written after the
#   attempt), failed (the label).
FEATURE_COLUMNS: list[str] = [
    "customer_id",
    "merchant_id",
    "area",
    "created_at",
    "promised_slot",
    "cod_amount",
    "n_items",
    "address_text",
]

HISTORY_COLUMNS = ["customer_prior_parcels", "customer_prior_fail_rate"]


# --------------------------------------------------------------------------
# History features (point in time)
# --------------------------------------------------------------------------
def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per customer: number of parcels strictly earlier than this row, and the
    share of those that failed (NaN if none). Ties are not 'earlier'. The result
    does not depend on row order, and the input is never modified."""
    out = df.copy()
    keys = ["customer_id", "created_at"]

    tmp = pd.DataFrame(
        {
            "customer_id": df["customer_id"].to_numpy(),
            "created_at": df["created_at"].to_numpy(),
            "_f": pd.to_numeric(df[TARGET], errors="coerce").fillna(0).to_numpy(dtype=float),
        }
    )
    # One row per (customer, timestamp): count and fails at that exact moment.
    g = (
        tmp.groupby(keys, sort=True, dropna=False)["_f"]
        .agg(_n="size", _fails="sum")
        .reset_index()
    )
    # Cumulative over time within a customer, minus the group's own rows
    # => only strictly earlier timestamps count.
    cum_n = g.groupby("customer_id", sort=False, dropna=False)["_n"].cumsum()
    cum_f = g.groupby("customer_id", sort=False, dropna=False)["_fails"].cumsum()
    g["_prior_n"] = (cum_n - g["_n"]).astype("int64")
    g["_prior_f"] = cum_f - g["_fails"]

    merged = tmp[keys].merge(g[keys + ["_prior_n", "_prior_f"]], on=keys, how="left")
    prior_n = merged["_prior_n"].to_numpy()
    prior_f = merged["_prior_f"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(prior_n > 0, prior_f / np.maximum(prior_n, 1), np.nan)

    out["customer_prior_parcels"] = prior_n.astype("int64")
    out["customer_prior_fail_rate"] = rate.astype(float)
    return out


# --------------------------------------------------------------------------
# Time split
# --------------------------------------------------------------------------
def time_split(
    df: pd.DataFrame, time_col: str = "created_at", test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < test_fraction < 1):
        raise ValueError("test_fraction must satisfy 0 < test_fraction < 1")
    n = len(df)
    n_test = math.ceil(n * test_fraction)
    if n == 0 or n_test >= n + 1:
        raise ValueError("not enough rows to split")
    times = df[time_col].sort_values(kind="mergesort")
    boundary = times.iloc[n - n_test]
    test = df[df[time_col] >= boundary]
    train = df[df[time_col] < boundary]
    return train, test


# --------------------------------------------------------------------------
# Pipeline (everything learnt from data lives inside)
# --------------------------------------------------------------------------
def extract_time_features(X) -> np.ndarray:
    timestamps = pd.to_datetime(
        pd.DataFrame(X).iloc[:, 0],
        errors="coerce"
    )

    return np.column_stack([
        timestamps.dt.hour.astype(float),
        timestamps.dt.dayofweek.astype(float)
    ])


def convert_to_text(X) -> pd.Series:
    if isinstance(X, pd.DataFrame):
        values = X.iloc[:, 0]
    else:
        values = np.asarray(X).ravel()

    return (
        pd.Series(values)
        .astype(object)
        .where(pd.Series(values).notna(), "")
        .astype(str)
        .reset_index(drop=True)
    )


def convert_to_object(X) -> pd.DataFrame:
    dataframe = pd.DataFrame(X)

    return dataframe.astype(object).where(
        dataframe.notna(),
        np.nan
    )


def convert_to_numeric(X) -> pd.DataFrame:
    return (
        pd.DataFrame(X)
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
    )


def build_pipeline(seed: int = 0) -> Pipeline:

    numeric_columns = ["cod_amount", "n_items"] + HISTORY_COLUMNS
    categorical_columns = ["area", "merchant_id", "promised_slot"]

    numeric_pipeline = make_pipeline(
        FunctionTransformer(convert_to_numeric),
        SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True
        ),
        StandardScaler()
    )

    time_pipeline = make_pipeline(
        FunctionTransformer(extract_time_features),
        SimpleImputer(
            strategy="median",
            keep_empty_features=True
        ),
        StandardScaler()
    )

    categorical_pipeline = make_pipeline(
        FunctionTransformer(convert_to_object),
        SimpleImputer(
            strategy="constant",
            fill_value="missing"
        ),
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=5
        )
    )

    text_pipeline = make_pipeline(
        FunctionTransformer(convert_to_text),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=3,
            max_features=3000,
            sublinear_tf=True
        )
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("time", time_pipeline, ["created_at"]),
            ("cat", categorical_pipeline, categorical_columns),
            ("text", text_pipeline, ["address_text"])
        ],
        remainder="drop"
    )

    classifier = LogisticRegression(
        C=0.3,
        max_iter=2000,
        random_state=seed
    )

    return Pipeline([
        ("pre", preprocessor),
        ("clf", classifier)
    ])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(model, test_df: pd.DataFrame) -> dict:
    y = test_df[TARGET]
    p = model.predict_proba(test_df)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "base_rate": float(np.mean(y)),
        "n": int(len(test_df)),
    }


def compare_to_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = 0) -> dict:
    if not all(c in train_df.columns for c in HISTORY_COLUMNS) or not all(
        c in test_df.columns for c in HISTORY_COLUMNS
    ):
        # Point-in-time history over past + present only; never uses future rows.
        both = add_history_features(pd.concat([train_df, test_df]))
        train_df, test_df = both.loc[train_df.index], both.loc[test_df.index]

    model = build_pipeline(seed).fit(train_df, train_df[TARGET])
    dummy = DummyClassifier(strategy="prior").fit(train_df, train_df[TARGET])
    m, d = evaluate(model, test_df), evaluate(dummy, test_df)
    return {"model": m, "dummy": d, "lift_pr_auc": m["pr_auc"] - d["pr_auc"]}
