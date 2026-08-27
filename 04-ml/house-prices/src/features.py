"""Preprocessing, built as a Pipeline rather than applied to the DataFrame.

This is the single most important file in the project, and the reason is
leakage. The tempting version looks like this:

    X[num] = SimpleImputer().fit_transform(X[num])     # WRONG
    X[num] = StandardScaler().fit_transform(X[num])
    scores = cross_val_score(model, X, y, cv=5)

The imputer's medians and the scaler's means were computed from **all** rows,
including the ones that will land in each validation fold. Every fold is then
scored on data whose statistics it already absorbed, and cross-validation
reports a number better than anything the model will achieve in production.

Wrapping preprocessing in a Pipeline fixes it structurally: `cross_val_score`
re-fits the whole pipeline inside each fold, so fold statistics come only from
that fold's training rows. There is a test asserting exactly this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL, NUMERIC


def numeric_pipeline(scale: bool = True) -> Pipeline:
    """Median impute, then optionally standardise.

    Median rather than mean because the numeric columns are right-skewed --
    a handful of 6,000 sqft houses drag the mean well above the typical value.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def categorical_pipeline() -> Pipeline:
    """Fill blanks with an explicit category, then one-hot encode.

    'missing' is its own level rather than the mode: a listing with no heating
    field may differ systematically from one that says 'gas', and collapsing
    the two throws that signal away.

    `handle_unknown='ignore'` matters at inference -- a neighbourhood absent
    from training encodes as all-zeros instead of raising.
    """
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Column-aware preprocessing: numeric and categorical handled separately."""
    return ColumnTransformer(
        [
            ("num", numeric_pipeline(scale), NUMERIC),
            ("cat", categorical_pipeline(), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable names for the transformed matrix, for importance plots."""
    return list(preprocessor.get_feature_names_out())


def add_derived(frame: pd.DataFrame) -> pd.DataFrame:
    """Ratios a linear model cannot construct for itself.

    Trees can approximate these by splitting repeatedly, but stating them
    directly is cheaper and makes the fitted coefficients interpretable.
    """
    out = frame.copy()
    out["rooms_total"] = out["bedrooms"] + out["bathrooms"]
    out["area_per_room"] = out["area_sqft"] / out["rooms_total"].replace(0, np.nan)
    out["lot_ratio"] = out["lot_sqft"] / out["area_sqft"]
    out["is_new"] = (out["age_years"] < 5).astype(float)
    return out


DERIVED = ["rooms_total", "area_per_room", "lot_ratio", "is_new"]


def build_preprocessor_with_derived(scale: bool = True) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", numeric_pipeline(scale), NUMERIC + DERIVED),
            ("cat", categorical_pipeline(), CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


__all__ = [
    "build_preprocessor", "build_preprocessor_with_derived",
    "numeric_pipeline", "categorical_pipeline",
    "feature_names", "add_derived", "DERIVED",
]
