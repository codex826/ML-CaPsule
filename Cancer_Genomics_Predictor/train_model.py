from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from data_utils import FEATURE_COLUMNS, TARGET_COLUMN, generate_synthetic_genomics_dataset, save_dataset


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = ARTIFACTS_DIR / "cancer_genomics_model.joblib"
DATASET_PATH = DATA_DIR / "synthetic_cancer_genomics.csv"


def train_and_save_model(random_state: int = 42) -> dict:
    dataset = generate_synthetic_genomics_dataset(random_state=random_state)
    save_dataset(dataset, DATASET_PATH)

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": report["macro avg"]["f1-score"],
        },
        "class_labels": sorted(y.unique().tolist()),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train_model() -> dict:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return train_and_save_model()


if __name__ == "__main__":
    bundle = train_and_save_model()
    print("Model trained successfully.")
    print(f"Accuracy: {bundle['metrics']['accuracy']:.3f}")
    print(f"Macro F1: {bundle['metrics']['macro_f1']:.3f}")
    print(f"Artifact: {MODEL_PATH}")
