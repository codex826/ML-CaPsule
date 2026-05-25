from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "tp53_expression",
    "brca1_expression",
    "egfr_expression",
    "kras_expression",
    "pik3ca_expression",
    "tumor_mutational_burden",
    "msi_score",
    "copy_number_instability",
    "patient_age",
    "smoking_index",
]

TARGET_COLUMN = "cancer_type"


def _class_frame(
    rng: np.random.Generator,
    label: str,
    size: int,
    means: dict[str, float],
    stds: dict[str, float],
) -> pd.DataFrame:
    data = {}
    for feature in FEATURE_COLUMNS:
        data[feature] = rng.normal(means[feature], stds[feature], size)

    frame = pd.DataFrame(data)
    frame["patient_age"] = frame["patient_age"].clip(18, 90).round().astype(int)
    frame["smoking_index"] = frame["smoking_index"].clip(0, 100)

    bounded_features = [
        "tp53_expression",
        "brca1_expression",
        "egfr_expression",
        "kras_expression",
        "pik3ca_expression",
        "tumor_mutational_burden",
        "msi_score",
        "copy_number_instability",
    ]
    for feature in bounded_features:
        frame[feature] = frame[feature].clip(0, 100)

    frame[TARGET_COLUMN] = label
    return frame


def generate_synthetic_genomics_dataset(
    n_samples_per_class: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    profiles = {
        "Breast-like": {
            "means": {
                "tp53_expression": 61,
                "brca1_expression": 79,
                "egfr_expression": 38,
                "kras_expression": 25,
                "pik3ca_expression": 74,
                "tumor_mutational_burden": 27,
                "msi_score": 21,
                "copy_number_instability": 63,
                "patient_age": 54,
                "smoking_index": 12,
            },
            "stds": {
                "tp53_expression": 9,
                "brca1_expression": 8,
                "egfr_expression": 10,
                "kras_expression": 8,
                "pik3ca_expression": 9,
                "tumor_mutational_burden": 8,
                "msi_score": 7,
                "copy_number_instability": 10,
                "patient_age": 8,
                "smoking_index": 10,
            },
        },
        "Lung-like": {
            "means": {
                "tp53_expression": 72,
                "brca1_expression": 34,
                "egfr_expression": 82,
                "kras_expression": 66,
                "pik3ca_expression": 46,
                "tumor_mutational_burden": 71,
                "msi_score": 19,
                "copy_number_instability": 58,
                "patient_age": 63,
                "smoking_index": 69,
            },
            "stds": {
                "tp53_expression": 11,
                "brca1_expression": 10,
                "egfr_expression": 9,
                "kras_expression": 10,
                "pik3ca_expression": 11,
                "tumor_mutational_burden": 12,
                "msi_score": 8,
                "copy_number_instability": 11,
                "patient_age": 9,
                "smoking_index": 13,
            },
        },
        "Colorectal-like": {
            "means": {
                "tp53_expression": 58,
                "brca1_expression": 31,
                "egfr_expression": 41,
                "kras_expression": 83,
                "pik3ca_expression": 59,
                "tumor_mutational_burden": 44,
                "msi_score": 76,
                "copy_number_instability": 42,
                "patient_age": 59,
                "smoking_index": 22,
            },
            "stds": {
                "tp53_expression": 10,
                "brca1_expression": 9,
                "egfr_expression": 10,
                "kras_expression": 8,
                "pik3ca_expression": 10,
                "tumor_mutational_burden": 10,
                "msi_score": 9,
                "copy_number_instability": 9,
                "patient_age": 8,
                "smoking_index": 11,
            },
        },
    }

    frames = []
    for label, config in profiles.items():
        frames.append(
            _class_frame(
                rng=rng,
                label=label,
                size=n_samples_per_class,
                means=config["means"],
                stds=config["stds"],
            )
        )

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return dataset


def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
