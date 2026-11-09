#!/usr/bin/env python3
"""
benchmark.py - Reproducible model benchmark for the used-car price model.

Regenerates the model comparison claimed in the README and writes a markdown
table to reports/benchmark.md. Every model sees the identical preprocessing
pipeline and the identical train/test split, so the comparison is fair.

Which models run depends on the data. Trees are depth-bounded so the whole
benchmark finishes in minutes, and any algorithm that cannot be run at the
given size is named in the report along with the reason, rather than dropped
in silence.

Requires scikit-learn >= 1.3 (TargetEncoder). Optional: xgboost, lightgbm, catboost.

Usage:
    python benchmark.py --data data/car_prices.csv --target sellingprice
    python benchmark.py --data data/car_prices.csv --target sellingprice --cv 0
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

try:
    from sklearn.preprocessing import TargetEncoder

    HAS_TARGET_ENCODER = True
except ImportError:  # scikit-learn < 1.3
    HAS_TARGET_ENCODER = False


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

#: Above this many training rows, KNN is excluded and the reason is reported.
#: Prediction compares every test row against every training row, so the cost
#: grows with their product rather than with either one.
KNN_ROW_BUDGET = 50_000

#: A column this close to unique is a record identifier, not a feature. Left in,
#: `vin` was target-encoded into 550,296 categories: ruinous to fit and empty of
#: signal, since a serial number says nothing about a car that its make, model
#: and year do not.
IDENTIFIER_UNIQUE_SHARE = 0.5


@dataclass(frozen=True)
class BenchmarkConfig:
    """Single source of truth for everything that affects the numbers."""

    data_path: Path
    target: str
    output_path: Path = Path("reports/benchmark.md")
    test_size: float = 0.20
    cv_folds: int = 5
    high_cardinality_threshold: int = 15
    max_rows: int | None = None
    random_state: int = 42
    drop_columns: tuple[str, ...] = field(default_factory=tuple)
    baseline_column: str | None = None


@dataclass
class ModelResult:
    """One row of the benchmark table."""

    name: str
    r2: float
    mae: float
    rmse: float
    fit_seconds: float
    cv_r2_mean: float | None = None
    cv_r2_std: float | None = None


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_dataset(cfg: BenchmarkConfig) -> tuple[pd.DataFrame, pd.Series]:
    """Read the raw CSV and return (features, target) with the target rows cleaned."""
    if not cfg.data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {cfg.data_path}. "
            "The CSV is gitignored - point --data at your local copy."
        )

    # The raw Kaggle export has unquoted commas inside a handful of `trim`
    # values, which makes those lines carry one field too many. Skipping them
    # matches what CARS.ipynb does; report the count so it stays visible.
    csv_rows = sum(1 for _ in cfg.data_path.open(encoding="utf-8", errors="replace")) - 1
    frame = pd.read_csv(cfg.data_path, low_memory=False, on_bad_lines="skip")
    malformed = csv_rows - len(frame)
    if malformed > 0:
        print(f"[data] skipped {malformed:,} malformed CSV rows")

    if cfg.target not in frame.columns:
        raise KeyError(
            f"Target column '{cfg.target}' not in dataset. "
            f"Available columns: {list(frame.columns)}"
        )

    # A row with no target teaches nothing and silently corrupts every metric.
    before = len(frame)
    frame = frame.dropna(subset=[cfg.target])
    dropped = before - len(frame)
    if dropped:
        print(f"[data] dropped {dropped:,} rows with a missing target")

    if cfg.drop_columns:
        frame = frame.drop(columns=[c for c in cfg.drop_columns if c in frame.columns])

    # Identifiers masquerade as high-cardinality categoricals: the encoder
    # accepts them, the fit takes hours, and the column cannot carry signal
    # because it names the row rather than describing it.
    identifiers = [
        column for column in frame.columns
        if column != cfg.target
        and frame[column].nunique(dropna=True) > len(frame) * IDENTIFIER_UNIQUE_SHARE
    ]
    if identifiers:
        for column in identifiers:
            share = frame[column].nunique(dropna=True) / len(frame)
            print(f"[data] dropping '{column}': {share:.1%} unique, an "
                  f"identifier rather than a feature")
        frame = frame.drop(columns=identifiers)

    if cfg.max_rows is not None and len(frame) > cfg.max_rows:
        frame = frame.sample(n=cfg.max_rows, random_state=cfg.random_state)
        print(f"[data] subsampled to {cfg.max_rows:,} rows for speed")

    target = frame[cfg.target].astype(float)
    features = frame.drop(columns=[cfg.target])

    print(f"[data] {len(features):,} rows x {features.shape[1]} features")
    return features, target


def split_feature_types(
    features: pd.DataFrame, threshold: int
) -> tuple[list[str], list[str], list[str]]:
    """Partition columns into numeric, low-cardinality and high-cardinality."""
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]

    low_card, high_card = [], []
    for column in categorical:
        (high_card if features[column].nunique() > threshold else low_card).append(column)

    print(
        f"[features] numeric={len(numeric)} "
        f"low_cardinality={len(low_card)} high_cardinality={len(high_card)}"
    )
    return numeric, low_card, high_card


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def build_preprocessor(
    features: pd.DataFrame, cfg: BenchmarkConfig
) -> ColumnTransformer:
    """
    One preprocessor for all nine models.

    Scaling is harmless for trees but mandatory for KNN and linear regression,
    so applying it once keeps the comparison honest.
    """
    numeric, low_card, high_card = split_feature_types(
        features, cfg.high_cardinality_threshold
    )

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    low_card_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )

    if HAS_TARGET_ENCODER:
        # target_type must be stated. Left on "auto", TargetEncoder asks
        # type_of_target(), which reports "multiclass" for any target whose
        # values are whole numbers - and a price in whole dollars is exactly
        # that, float dtype or not. It then encodes one column per category per
        # class: with 773 distinct prices in the sample, nine categorical
        # columns became 6,957. That is what made this benchmark take hours and
        # what drove linear regression to an R2 of -150.
        high_card_encoder = TargetEncoder(
            target_type="continuous", random_state=cfg.random_state
        )
    else:
        print("[warn] scikit-learn < 1.3: falling back to ordinal encoding")
        high_card_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

    high_card_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("encode", high_card_encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("low_card", low_card_pipe, low_card),
            ("high_card", high_card_pipe, high_card),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def build_model_registry(
    random_state: int, n_train: int
) -> tuple[dict[str, object], list[tuple[str, str]]]:
    """The algorithms to benchmark, and the ones this dataset is too big for.

    Every tree here carries a depth bound. Without one, a single tree fitted on
    450k rows grows until each leaf holds one sample: it memorises the training
    set, costs gigabytes, and tells you nothing a bounded tree does not. Left
    unbounded with 5-fold CV - which fits every model six times - this benchmark
    ran for nine and a half hours on a quarter of the data without finishing.
    The bounds below are what make the comparison reproducible; they are not
    tuned, and no model gets a bound another model does not.
    """
    skipped: list[tuple[str, str]] = []
    registry: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=14, min_samples_leaf=5, random_state=random_state
        ),
        "AdaBoost": AdaBoostRegressor(random_state=random_state),
        # sklearn's exact GradientBoostingRegressor is single-threaded and scans
        # every split candidate, so it does not finish at this scale. The
        # histogram implementation is the same algorithm on binned features and
        # is what sklearn now recommends above ~10k rows.
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, random_state=random_state
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            random_state=random_state, n_jobs=-1,
        ),
    }

    # KNN stores the training set and compares every test row against all of it,
    # so prediction cost is the product of the two. That is a property of the
    # algorithm, not a configuration to fix, and it is worth reporting rather
    # than hiding: on the full dataset this is ~5e10 distance computations.
    if n_train <= KNN_ROW_BUDGET:
        registry["KNN"] = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    else:
        skipped.append((
            "KNN",
            f"prediction is quadratic in dataset size; {n_train:,} training "
            f"rows exceeds the {KNN_ROW_BUDGET:,}-row budget. Re-run with "
            f"--max-rows {KNN_ROW_BUDGET} to include it.",
        ))

    try:
        from xgboost import XGBRegressor

        registry["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    except ImportError:
        print("[warn] xgboost not installed - skipping")

    try:
        from lightgbm import LGBMRegressor

        registry["LightGBM"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("[warn] lightgbm not installed - skipping")

    try:
        from catboost import CatBoostRegressor

        registry["CatBoost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=0,
        )
    except ImportError:
        print("[warn] catboost not installed - skipping")

    return registry, skipped


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_model(
    name: str,
    estimator: object,
    preprocessor: ColumnTransformer,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    cfg: BenchmarkConfig,
) -> ModelResult:
    """Fit one model on the shared split and measure held-out performance."""
    x_train, x_test, y_train, y_test = splits
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    predictions = pipeline.predict(x_test)

    cv_mean = cv_std = None
    if cfg.cv_folds > 1:
        folds = KFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )
        scores = cross_val_score(
            pipeline, x_train, y_train, cv=folds, scoring="r2", n_jobs=1
        )
        cv_mean, cv_std = float(scores.mean()), float(scores.std())

    return ModelResult(
        name=name,
        r2=float(r2_score(y_test, predictions)),
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        fit_seconds=fit_seconds,
        cv_r2_mean=cv_mean,
        cv_r2_std=cv_std,
    )


def run_benchmark(
    cfg: BenchmarkConfig,
) -> tuple[list[ModelResult], list[tuple[str, str]]]:
    """Run every registered model against one shared split."""
    features, target = load_dataset(cfg)
    preprocessor = build_preprocessor(features, cfg)

    splits = train_test_split(
        features, target, test_size=cfg.test_size, random_state=cfg.random_state
    )
    print(f"[split] train={len(splits[0]):,} test={len(splits[1]):,}")
    if cfg.cv_folds > 1:
        print(f"[cost] {cfg.cv_folds}-fold CV fits every model "
              f"{cfg.cv_folds + 1} times")
    print()

    registry, skipped = build_model_registry(cfg.random_state, len(splits[0]))
    for name, reason in skipped:
        print(f"[skip] {name}: {reason}")
    if skipped:
        print()

    results: list[ModelResult] = []

    # A model is only worth its complexity if it beats the obvious thing. Where
    # one column is already an estimate of the target - here `mmr`, a published
    # wholesale appraisal of the same vehicle - copying it is the obvious thing,
    # and any model that cannot beat it has earned nothing.
    if cfg.baseline_column:
        x_test, y_test = splits[1], splits[3]
        if cfg.baseline_column not in x_test.columns:
            print(f"[warn] baseline column '{cfg.baseline_column}' not in the "
                  f"features; skipping the baseline")
        else:
            guess = x_test[cfg.baseline_column].astype(float)
            usable = guess.notna()
            results.append(ModelResult(
                name=f"(no model) copy `{cfg.baseline_column}`",
                r2=float(r2_score(y_test[usable], guess[usable])),
                mae=float(mean_absolute_error(y_test[usable], guess[usable])),
                rmse=float(np.sqrt(mean_squared_error(y_test[usable], guess[usable]))),
                fit_seconds=0.0,
            ))
            print(f"[base] copy `{cfg.baseline_column}`  R2={results[0].r2:.4f}  "
                  f"MAE={results[0].mae:,.0f}  (no model at all)\n")
    for name, estimator in registry.items():
        print(f"[fit] {name} ...", end=" ", flush=True)
        try:
            result = evaluate_model(name, estimator, preprocessor, splits, cfg)
        except Exception as error:  # a broken model must not kill the whole run
            print(f"FAILED: {error}")
            skipped.append((name, f"failed during fit: {error}"))
            continue
        print(f"R2={result.r2:.4f}  MAE={result.mae:,.0f}  ({result.fit_seconds:.1f}s)")
        results.append(result)

    return sorted(results, key=lambda r: r.r2, reverse=True), skipped


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def results_to_markdown(
    results: list[ModelResult],
    cfg: BenchmarkConfig,
    skipped: list[tuple[str, str]] | None = None,
) -> str:
    """Render the results as a README-ready markdown table."""
    show_cv = any(r.cv_r2_mean is not None for r in results)
    # The baseline shares the table but is not one of the models being compared.
    model_count = sum(1 for r in results if not r.name.startswith("(no model)"))

    header = "| Model | R² (test) | MAE | RMSE | Fit (s) |"
    divider = "| --- | ---: | ---: | ---: | ---: |"
    if show_cv:
        # Append the column; do not strip the closing pipe first, or the CV
        # column merges into "Fit (s)" and the table stops rendering as one.
        header += f" {cfg.cv_folds}-fold CV R² |"
        divider += " ---: |"

    lines = [
        "## Model Benchmark",
        "",
        f"All {model_count} models share one preprocessing pipeline and one "
        f"train/test split ({int((1 - cfg.test_size) * 100)}/"
        f"{int(cfg.test_size * 100)}, `random_state={cfg.random_state}`).",
        # The reproduce line has to carry the flags this run actually used.
        # Printing the bare command invites someone to run something else and
        # wonder why the numbers moved.
        "Reproduce with `python benchmark.py --data <path> "
        f"--target {cfg.target}"
        + (f" --drop {' '.join(cfg.drop_columns)}" if cfg.drop_columns else "")
        + (f" --baseline {cfg.baseline_column}" if cfg.baseline_column else "")
        + (f" --max-rows {cfg.max_rows}" if cfg.max_rows else "")
        + f" --cv {cfg.cv_folds}`.",
        "",
        header,
        divider,
    ]

    for result in results:
        row = (
            f"| {result.name} | {result.r2:.4f} | {result.mae:,.0f} | "
            f"{result.rmse:,.0f} | {result.fit_seconds:.1f} |"
        )
        if show_cv:
            row += (
                f" {result.cv_r2_mean:.4f} ± {result.cv_r2_std:.4f} |"
                if result.cv_r2_mean is not None
                else " n/a |"
            )
        lines.append(row)

    # The baseline is a reference, not a candidate. If it outranks every model,
    # that is the headline and it should be stated rather than hidden by
    # quietly crowning whatever came second.
    models = [r for r in results if not r.name.startswith("(no model)")]
    baseline = next((r for r in results if r.name.startswith("(no model)")), None)

    if models:
        best = models[0]
        lines += [
            "",
            f"**Selected model: {best.name}** - R² {best.r2:.4f}, "
            f"MAE {best.mae:,.0f} on the held-out test set.",
        ]
        if baseline is not None:
            delta_r2 = best.r2 - baseline.r2
            delta_mae = baseline.mae - best.mae
            verdict = (
                f"It beats the no-model baseline by {delta_r2:+.4f} R² and "
                f"{delta_mae:,.0f} MAE."
                if delta_r2 > 0 else
                f"**It does not beat the no-model baseline**, which scores "
                f"{baseline.r2:.4f} R² and {baseline.mae:,.0f} MAE. The "
                f"headline number here measures the baseline column, not the "
                f"model."
            )
            lines += ["", verdict]

    # A model left out of the table is a hole in the comparison. Saying which
    # and why is the difference between a benchmark and a highlight reel.
    if skipped:
        lines += ["", "### Not benchmarked", ""]
        lines += [f"- **{name}** - {reason}" for name, reason in skipped]

    lines += [
        "",
        "Every tree-based model is depth-bounded so the run is reproducible in "
        "minutes rather than hours; the bounds are not tuned, and no model "
        "receives one another does not. `HistGradientBoosting` stands in for "
        "sklearn's exact `GradientBoostingRegressor`, which is single-threaded "
        "and does not finish at this scale.",
    ]

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="path to the CSV")
    parser.add_argument("--target", type=str, required=True, help="target column name")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.md"))
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=5, help="CV folds; 0 disables")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop", nargs="*", default=[], help="columns to exclude")
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="column that is already an estimate of the target; it is copied "
             "as a prediction and reported as the no-model reference every "
             "model has to beat",
    )
    args = parser.parse_args(argv)

    return BenchmarkConfig(
        data_path=args.data,
        target=args.target,
        output_path=args.output,
        test_size=args.test_size,
        cv_folds=args.cv,
        max_rows=args.max_rows,
        random_state=args.seed,
        drop_columns=tuple(args.drop),
        baseline_column=args.baseline,
    )


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)
    results, skipped = run_benchmark(cfg)

    if not results:
        print("\n[error] no model completed successfully")
        return 1

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(
        results_to_markdown(results, cfg, skipped), encoding="utf-8"
    )

    print(f"\n[done] {len(results)} models benchmarked -> {cfg.output_path}")
    print("Paste the contents of that file into README.md under '## Results'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
