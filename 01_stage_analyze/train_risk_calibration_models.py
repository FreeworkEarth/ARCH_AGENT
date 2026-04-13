#!/usr/bin/env python3
"""
train_risk_calibration_models.py
================================

Train simple time-aware calibration models for file-level risk prediction.

Models:
  1. linear_nnls     : non-negative linear model on core risk signals
  2. quadratic_nnls  : non-negative model on core signals + squares + pairwise interactions

The split is chronological by cutoff revision:
  older cutoffs -> train
  newest cutoffs -> holdout
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr


CORE_FEATURES = [
    "bug_churn_total",
    "total_churn",
    "anti_pattern_count",
    "hotspot_fanin_score",
    "scc_membership_count",
    "co_change_without_dep",
]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    print(f"  Written: {path}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def _chronological_split(df: pd.DataFrame, holdout_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    cutoffs = sorted(df["cutoff_revision_number"].unique(), reverse=True)  # oldest -> newest
    split_idx = max(1, int(round(len(cutoffs) * (1.0 - holdout_frac))))
    split_idx = min(split_idx, len(cutoffs) - 1)
    train_cutoffs = cutoffs[:split_idx]
    test_cutoffs = cutoffs[split_idx:]
    train_df = df[df["cutoff_revision_number"].isin(train_cutoffs)].copy()
    test_df = df[df["cutoff_revision_number"].isin(test_cutoffs)].copy()
    return train_df, test_df, train_cutoffs, test_cutoffs


def _fit_minmax(train_df: pd.DataFrame, feature_names: List[str]) -> Dict[str, Tuple[float, float]]:
    scaler: Dict[str, Tuple[float, float]] = {}
    for f in feature_names:
        vmin = float(train_df[f].min())
        vmax = float(train_df[f].max())
        scaler[f] = (vmin, vmax)
    return scaler


def _apply_minmax(df: pd.DataFrame, scaler: Dict[str, Tuple[float, float]], feature_names: List[str]) -> np.ndarray:
    cols = []
    for f in feature_names:
        vmin, vmax = scaler[f]
        vals = df[f].astype(float).to_numpy()
        span = vmax - vmin
        if span <= 0:
            cols.append(np.zeros_like(vals, dtype=float))
        else:
            cols.append((vals - vmin) / span)
    return np.column_stack(cols)


def _expand_quadratic(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    cols = [X]
    names = list(feature_names)
    # squares
    squares = X ** 2
    cols.append(squares)
    names.extend([f"{f}^2" for f in feature_names])
    # pairwise interactions
    inter_cols = []
    inter_names = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            inter_cols.append((X[:, i] * X[:, j])[:, None])
            inter_names.append(f"{feature_names[i]}*{feature_names[j]}")
    if inter_cols:
        cols.append(np.hstack(inter_cols))
        names.extend(inter_names)
    return np.hstack(cols), names


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        rho = float(spearmanr(y_true, y_pred).statistic)
    else:
        rho = 0.0
    # precision@top-k on future bug files
    pos_count = int(np.sum(y_true > 0))
    k = max(10, min(50, pos_count if pos_count > 0 else 10))
    order = np.argsort(-y_pred)
    top_idx = order[:k]
    precision_at_k = float(np.mean((y_true[top_idx] > 0).astype(float)))
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman_rho": rho,
        "precision_at_k": precision_at_k,
        "k": int(k),
    }


def _fit_and_predict_nnls(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if alpha > 0:
        reg_X = math.sqrt(alpha) * np.eye(X_train.shape[1])
        reg_y = np.zeros(X_train.shape[1], dtype=float)
        fit_X = np.vstack([X_train, reg_X])
        fit_y = np.concatenate([y_train_log, reg_y])
    else:
        fit_X = X_train
        fit_y = y_train_log
    coef, _ = nnls(fit_X, fit_y)
    train_pred_log = X_train @ coef
    test_pred_log = X_test @ coef
    return coef, np.expm1(train_pred_log), np.expm1(test_pred_log)


def _top_weights(names: List[str], coef: np.ndarray, top_n: int = 15) -> List[Dict[str, Any]]:
    pairs = [(n, float(w)) for n, w in zip(names, coef) if float(w) > 0]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [{"feature": n, "weight": round(w, 8)} for n, w in pairs[:top_n]]


def train_models(dataset_path: Path, *, holdout_frac: float, target_col: str) -> Dict[str, Any]:
    df = _load_dataset(dataset_path)
    required = ["cutoff_revision_number", "file", target_col] + CORE_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {missing}")

    train_df, test_df, train_cutoffs, test_cutoffs = _chronological_split(df, holdout_frac)
    scaler = _fit_minmax(train_df, CORE_FEATURES)

    X_train_base = _apply_minmax(train_df, scaler, CORE_FEATURES)
    X_test_base = _apply_minmax(test_df, scaler, CORE_FEATURES)
    y_train = train_df[target_col].astype(float).to_numpy()
    y_test = test_df[target_col].astype(float).to_numpy()
    y_train_log = np.log1p(y_train)

    # Linear
    linear_alpha = 0.01
    quad_alpha = 0.5

    linear_coef, linear_train_pred, linear_test_pred = _fit_and_predict_nnls(
        X_train_base, y_train_log, X_test_base, alpha=linear_alpha
    )

    # Quadratic interactions
    X_train_quad, quad_names = _expand_quadratic(X_train_base, CORE_FEATURES)
    X_test_quad, _ = _expand_quadratic(X_test_base, CORE_FEATURES)
    quad_coef, quad_train_pred, quad_test_pred = _fit_and_predict_nnls(
        X_train_quad, y_train_log, X_test_quad, alpha=quad_alpha
    )

    linear_summary = {
        "model": "linear_nnls",
        "l2_alpha": linear_alpha,
        "features": CORE_FEATURES,
        "weights": {k: round(float(v), 8) for k, v in zip(CORE_FEATURES, linear_coef)},
        "train_metrics": _metrics(y_train, linear_train_pred),
        "holdout_metrics": _metrics(y_test, linear_test_pred),
        "top_weights": _top_weights(CORE_FEATURES, linear_coef),
    }

    quad_summary = {
        "model": "quadratic_nnls",
        "l2_alpha": quad_alpha,
        "feature_count": len(quad_names),
        "train_metrics": _metrics(y_train, quad_train_pred),
        "holdout_metrics": _metrics(y_test, quad_test_pred),
        "top_weights": _top_weights(quad_names, quad_coef),
    }

    pred_rows: List[Dict[str, Any]] = []
    for source_df, preds_linear, preds_quad, split_name in [
        (train_df, linear_train_pred, quad_train_pred, "train"),
        (test_df, linear_test_pred, quad_test_pred, "holdout"),
    ]:
        for idx, row in enumerate(source_df.itertuples(index=False)):
            pred_rows.append(
                {
                    "split": split_name,
                    "cutoff_revision_number": int(getattr(row, "cutoff_revision_number")),
                    "file": getattr(row, "file"),
                    "actual_target": float(getattr(row, target_col)),
                    "linear_pred": round(float(preds_linear[idx]), 6),
                    "quadratic_pred": round(float(preds_quad[idx]), 6),
                }
            )

    return {
        "meta": {
            "dataset_path": str(dataset_path),
            "target": target_col,
            "holdout_frac": holdout_frac,
            "train_cutoffs": train_cutoffs,
            "holdout_cutoffs": test_cutoffs,
            "train_rows": int(len(train_df)),
            "holdout_rows": int(len(test_df)),
        },
        "linear_nnls": linear_summary,
        "quadratic_nnls": quad_summary,
        "predictions": pred_rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train simple linear/nonlinear calibration models on the risk dataset.")
    ap.add_argument("dataset", help="Path to calibration dataset JSON or CSV")
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="Newest cutoff fraction reserved for holdout (default: 0.2)")
    ap.add_argument("--target", default="future_bug_churn_3w", help="Target column (default: future_bug_churn_3w)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    out_dir = dataset_path.parent
    result = train_models(dataset_path, holdout_frac=args.holdout_frac, target_col=args.target)
    _write_json(out_dir / "risk_model_training_summary.json", {k: v for k, v in result.items() if k != "predictions"})
    pred_rows = result["predictions"]
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "risk_model_predictions.csv", index=False)
    print(f"  Written: {out_dir / 'risk_model_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
