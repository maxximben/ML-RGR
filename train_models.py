from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cars.csv"
MODELS_DIR = ROOT / "models"
TARGET = "Price(euro)"
RANDOM_STATE = 42


def build_models() -> dict[str, object]:
    return {
        "ML1_polynomial_ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=25.0, random_state=RANDOM_STATE)),
            ]
        ),
        "ML2_gradient_boosting": GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.06,
            max_depth=4,
            subsample=0.85,
            random_state=RANDOM_STATE,
        ),
        "ML4_bagging": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=18, min_samples_leaf=3, random_state=RANDOM_STATE),
            n_estimators=80,
            max_samples=0.85,
            max_features=0.9,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "ML5_stacking": StackingRegressor(
            estimators=[
                ("ridge", Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=10.0))])),
                ("rf", RandomForestRegressor(n_estimators=80, max_depth=18, n_jobs=-1, random_state=RANDOM_STATE)),
                ("gbr", GradientBoostingRegressor(n_estimators=120, learning_rate=0.07, max_depth=3, random_state=RANDOM_STATE)),
            ],
            final_estimator=Ridge(alpha=5.0),
            cv=3,
            n_jobs=-1,
        ),
        "ML6_mlp_neural_network": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=0.001,
                        batch_size=256,
                        learning_rate_init=0.001,
                        max_iter=220,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def regression_report(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = np.asarray(model.predict(x_test)).clip(min=0)
    return {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    }


def save_pickle(model: object, path: Path) -> None:
    with path.open("wb") as file:
        pickle.dump(model, file)


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    x = df.drop(columns=[TARGET])
    y = df[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    metrics: dict[str, dict[str, float | str]] = {}
    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        save_pickle(model, MODELS_DIR / f"{name}.pkl")
        metrics[name] = regression_report(model, x_test, y_test) | {"file": f"{name}.pkl"}
        print(f"{name}: R2={metrics[name]['r2']:.4f}")

    print("Training ML3_catboost...")
    cat_model = CatBoostRegressor(
        iterations=700,
        depth=8,
        learning_rate=0.06,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    cat_model.fit(x_train, y_train)
    cat_model.save_model(str(MODELS_DIR / "ML3_catboost.cbm"))
    metrics["ML3_catboost"] = regression_report(cat_model, x_test, y_test) | {"file": "ML3_catboost.cbm"}
    print(f"ML3_catboost: R2={metrics['ML3_catboost']['r2']:.4f}")

    metadata = {
        "target": TARGET,
        "features": list(x.columns),
        "rows": int(len(df)),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "metrics": dict(sorted(metrics.items(), key=lambda item: item[1]["r2"], reverse=True)),
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved models and metadata to models/")


if __name__ == "__main__":
    main()
