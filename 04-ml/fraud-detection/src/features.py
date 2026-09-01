"""Preprocessing, with the imbalance kept in mind.

Same Pipeline discipline as everywhere else -- preprocessing is fitted inside
each CV fold, never on the whole dataset.

One imbalance-specific note: **no resampling happens here.** SMOTE and random
oversampling are popular and mostly counterproductive for this problem. They
distort the base rate, which means predicted probabilities no longer mean what
they say, and a threshold chosen on resampled data does not transfer to
production traffic. `class_weight='balanced'` achieves the same re-weighting
without inventing rows or corrupting calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC

# Amounts span 0.50 to 9,000 with a long right tail. log1p compresses it so a
# linear model is not dominated by a handful of large transactions.
LOG_COLUMNS = ["amount", "avg_amount_30d", "distance_from_home_km", "amount_vs_avg"]
PLAIN_COLUMNS = [c for c in NUMERIC if c not in LOG_COLUMNS]


def log_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )


def plain_pipeline() -> Pipeline:
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )


def categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Log-compress the skewed money columns, standardise the rest, one-hot the categoricals."""
    if not scale:
        # Trees split on raw values; scaling and logging buy them nothing.
        return ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), NUMERIC),
                ("cat", categorical_pipeline(), CATEGORICAL),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    return ColumnTransformer(
        [
            ("log", log_pipeline(), LOG_COLUMNS),
            ("num", plain_pipeline(), PLAIN_COLUMNS),
            ("cat", categorical_pipeline(), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


__all__ = [
    "build_preprocessor", "feature_names",
    "log_pipeline", "plain_pipeline", "categorical_pipeline",
    "LOG_COLUMNS", "PLAIN_COLUMNS",
]
