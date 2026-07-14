"""Classical ML models: Logistic Regression, Random Forest, XGBoost."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model(model_type: str = "xgboost", class_weight: str = "balanced", seed: int = 42):
    model_type = model_type.lower()
    if model_type in ("logistic", "lr", "logreg"):
        clf = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight if class_weight != "none" else None,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type in ("rf", "random_forest"):
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weight if class_weight != "none" else None,
            random_state=seed,
            n_jobs=-1,
        )

    if model_type in ("xgb", "xgboost"):
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is required for model_type=xgboost") from e

        # scale_pos_weight set later in train if needed
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Any:
    """Fit model; for XGBoost set scale_pos_weight from train class ratio."""
    name = type(model).__name__.lower()
    if "xgb" in name:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = float(neg / max(pos, 1))
        model.set_params(scale_pos_weight=spw)
        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            # early stopping if supported
            try:
                model.set_params(early_stopping_rounds=30)
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False
            except Exception:
                pass
        model.fit(X_train, y_train, **fit_kwargs)
    else:
        model.fit(X_train, y_train)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # decision_function fallback
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X).astype(np.float64)
