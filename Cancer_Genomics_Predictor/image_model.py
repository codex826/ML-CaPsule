from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
IMAGE_MODEL_PATH = ARTIFACTS_DIR / "cancer_scan_model.joblib"
SCAN_DATASET_PATH = DATA_DIR / "synthetic_scan_features.csv"

SCAN_FEATURE_COLUMNS = [
    "mean_intensity",
    "std_intensity",
    "center_brightness",
    "left_right_asymmetry",
    "upper_lower_difference",
    "edge_strength",
    "hotspot_fraction",
    "center_hotspot_fraction",
    "intensity_p95",
]

SCAN_FEATURE_LABELS = {
    "mean_intensity": "overall intensity",
    "std_intensity": "texture variation",
    "center_brightness": "central brightness",
    "left_right_asymmetry": "left-right asymmetry",
    "upper_lower_difference": "upper-lower contrast",
    "edge_strength": "edge strength",
    "hotspot_fraction": "bright hotspot fraction",
    "center_hotspot_fraction": "central hotspot fraction",
    "intensity_p95": "high-intensity tail",
}

SCAN_CLASS_DESCRIPTIONS = {
    "Normal-like Scan": (
        "The uploaded image most closely matches the normal-like pattern in this demo. "
        "That usually means the scan has smoother texture, lower focal hotspot activity, and lower asymmetry."
    ),
    "Localized Nodule-like Pattern": (
        "The uploaded image most closely matches a localized nodule-like pattern in this demo. "
        "This type is usually associated with a small focal bright region and moderate asymmetry."
    ),
    "Mass-like Pattern": (
        "The uploaded image most closely matches a mass-like pattern in this demo. "
        "This class is typically driven by stronger bright clusters, denser edges, and more pronounced asymmetry."
    ),
}


def _base_scan_canvas(rng: np.random.Generator, size: int = 128) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    canvas = rng.normal(0.62, 0.035, (size, size))

    left_lung = ((x - size * 0.36) / (size * 0.18)) ** 2 + ((y - size * 0.56) / (size * 0.28)) ** 2 <= 1
    right_lung = ((x - size * 0.64) / (size * 0.18)) ** 2 + ((y - size * 0.56) / (size * 0.28)) ** 2 <= 1
    canvas[left_lung] -= 0.18
    canvas[right_lung] -= 0.18

    spine = np.exp(-((x - size * 0.5) ** 2) / (2 * (size * 0.028) ** 2))
    canvas += 0.1 * spine

    rib_pattern = 0.028 * np.sin(y / 4.0) * np.exp(-((x - size * 0.5) ** 2) / (2 * (size * 0.36) ** 2))
    canvas += rib_pattern

    canvas += 0.04 * np.exp(-((y - size * 0.18) ** 2) / (2 * (size * 0.06) ** 2))
    return np.clip(canvas, 0, 1)


def _add_blob(
    canvas: np.ndarray,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
    strength: float,
) -> np.ndarray:
    size = canvas.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    blob = np.exp(
        -(
            ((x - center_x) ** 2) / (2 * radius_x ** 2)
            + ((y - center_y) ** 2) / (2 * radius_y ** 2)
        )
    )
    canvas += strength * blob
    return canvas


def _render_scan_image(label: str, rng: np.random.Generator, size: int = 128) -> Image.Image:
    canvas = _base_scan_canvas(rng, size=size)

    if label == "Localized Nodule-like Pattern":
        x_center = rng.choice([size * 0.35, size * 0.65]) + rng.normal(0, size * 0.035)
        y_center = size * 0.55 + rng.normal(0, size * 0.06)
        canvas = _add_blob(
            canvas,
            center_x=x_center,
            center_y=y_center,
            radius_x=size * 0.045,
            radius_y=size * 0.045,
            strength=0.42,
        )

    elif label == "Mass-like Pattern":
        x_center = rng.choice([size * 0.34, size * 0.66]) + rng.normal(0, size * 0.045)
        y_center = size * 0.53 + rng.normal(0, size * 0.05)
        canvas = _add_blob(
            canvas,
            center_x=x_center,
            center_y=y_center,
            radius_x=size * 0.09,
            radius_y=size * 0.11,
            strength=0.38,
        )
        canvas = _add_blob(
            canvas,
            center_x=x_center + rng.normal(0, size * 0.03),
            center_y=y_center + rng.normal(0, size * 0.03),
            radius_x=size * 0.05,
            radius_y=size * 0.06,
            strength=0.22,
        )

    image = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), mode="L")

    if label == "Mass-like Pattern":
        image = image.filter(ImageFilter.GaussianBlur(radius=1.4))
    else:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.9))

    return image


def extract_image_features(image: Image.Image) -> dict[str, float]:
    grayscale = image.convert("L").resize((128, 128))
    arr = np.asarray(grayscale, dtype=np.float32) / 255.0

    center = arr[32:96, 32:96]
    left_half = arr[:, :64]
    right_half = np.fliplr(arr[:, 64:])
    upper_half = arr[:64, :]
    lower_half = arr[64:, :]

    grad_y, grad_x = np.gradient(arr)
    edge_map = np.sqrt(grad_x ** 2 + grad_y ** 2)

    features = {
        "mean_intensity": float(arr.mean()),
        "std_intensity": float(arr.std()),
        "center_brightness": float(center.mean()),
        "left_right_asymmetry": float(np.mean(np.abs(left_half - right_half))),
        "upper_lower_difference": float(abs(upper_half.mean() - lower_half.mean())),
        "edge_strength": float(edge_map.mean()),
        "hotspot_fraction": float(np.mean(arr > 0.76)),
        "center_hotspot_fraction": float(np.mean(center > 0.76)),
        "intensity_p95": float(np.percentile(arr, 95)),
    }
    return features


def _build_scan_feature_dataset(
    n_samples_per_class: int = 320,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    labels = list(SCAN_CLASS_DESCRIPTIONS.keys())

    rows: list[dict[str, float | str]] = []
    for label in labels:
        for _ in range(n_samples_per_class):
            image = _render_scan_image(label, rng)
            features = extract_image_features(image)
            features["scan_pattern"] = label
            rows.append(features)

    frame = pd.DataFrame(rows)
    frame = frame.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return frame


def train_and_save_image_model(random_state: int = 42) -> dict:
    dataset = _build_scan_feature_dataset(random_state=random_state)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(SCAN_DATASET_PATH, index=False)

    X = dataset[SCAN_FEATURE_COLUMNS]
    y = dataset["scan_pattern"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=220,
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
        "feature_columns": SCAN_FEATURE_COLUMNS,
        "feature_labels": SCAN_FEATURE_LABELS,
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": report["macro avg"]["f1-score"],
        },
        "class_descriptions": SCAN_CLASS_DESCRIPTIONS,
        "reference_stats": {
            "mean": X.mean().to_dict(),
            "std": X.std().replace(0, 1e-6).to_dict(),
        },
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, IMAGE_MODEL_PATH)
    return bundle


def load_or_train_image_model() -> dict:
    if IMAGE_MODEL_PATH.exists():
        return joblib.load(IMAGE_MODEL_PATH)
    return train_and_save_image_model()


def summarize_scan_prediction(bundle: dict, image: Image.Image) -> dict:
    features = extract_image_features(image)
    frame = pd.DataFrame([features], columns=SCAN_FEATURE_COLUMNS)
    model = bundle["model"]
    probabilities = model.predict_proba(frame)[0]
    prediction = model.predict(frame)[0]

    global_means = bundle["reference_stats"]["mean"]
    global_stds = bundle["reference_stats"]["std"]

    ranked_signals = sorted(
        SCAN_FEATURE_COLUMNS,
        key=lambda feature: abs((features[feature] - global_means[feature]) / global_stds[feature]),
        reverse=True,
    )

    signal_lines = []
    for feature in ranked_signals[:3]:
        label = bundle["feature_labels"][feature]
        direction = "elevated" if features[feature] >= global_means[feature] else "reduced"
        signal_lines.append(f"{label} is {direction}")

    class_probabilities = [
        {"label": label, "probability": float(prob)}
        for label, prob in sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    explanation = bundle["class_descriptions"][prediction]
    return {
        "predicted_class": prediction,
        "confidence": float(max(probabilities)),
        "probabilities": class_probabilities,
        "signal_lines": signal_lines,
        "explanation": explanation,
    }
