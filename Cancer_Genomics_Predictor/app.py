from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request, session
from werkzeug.utils import secure_filename

from chatbot import respond
from data_utils import FEATURE_COLUMNS
from image_model import SCAN_DATASET_PATH, load_or_train_image_model, summarize_scan_prediction
from train_model import DATASET_PATH, load_or_train_model


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "cancer-genomics-demo-secret")
model_bundle = load_or_train_model()
scan_model_bundle = load_or_train_image_model()
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_META = {
    "tp53_expression": {"label": "TP53 Expression", "min": 0, "max": 100, "step": "0.1", "default": 60},
    "brca1_expression": {"label": "BRCA1 Expression", "min": 0, "max": 100, "step": "0.1", "default": 55},
    "egfr_expression": {"label": "EGFR Expression", "min": 0, "max": 100, "step": "0.1", "default": 50},
    "kras_expression": {"label": "KRAS Expression", "min": 0, "max": 100, "step": "0.1", "default": 45},
    "pik3ca_expression": {"label": "PIK3CA Expression", "min": 0, "max": 100, "step": "0.1", "default": 58},
    "tumor_mutational_burden": {"label": "Tumor Mutational Burden", "min": 0, "max": 100, "step": "0.1", "default": 35},
    "msi_score": {"label": "MSI Score", "min": 0, "max": 100, "step": "0.1", "default": 25},
    "copy_number_instability": {"label": "Copy Number Instability", "min": 0, "max": 100, "step": "0.1", "default": 50},
    "patient_age": {"label": "Patient Age", "min": 18, "max": 90, "step": "1", "default": 56},
    "smoking_index": {"label": "Smoking Index", "min": 0, "max": 100, "step": "0.1", "default": 20},
}

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

GENOMICS_CONTEXT_KEY = "last_genomics_prediction"
GENOMICS_INPUTS_KEY = "last_genomics_inputs"
IMAGE_CONTEXT_KEY = "last_image_prediction"


def _parse_form_values(form: dict) -> dict[str, float]:
    values = {}
    for feature in FEATURE_COLUMNS:
        values[feature] = float(form.get(feature, FEATURE_META[feature]["default"]))
    return values


def _predict_from_values(values: dict[str, float]) -> dict:
    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    model = model_bundle["model"]
    probabilities = model.predict_proba(frame)[0]
    prediction = model.predict(frame)[0]

    ranked = sorted(
        zip(model_bundle["feature_columns"], model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    top_features = [name for name, _ in ranked[:3]]

    class_probabilities = [
        {"label": label, "probability": float(prob)}
        for label, prob in sorted(
            zip(model.classes_, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    return {
        "predicted_class": prediction,
        "confidence": float(max(probabilities)),
        "top_features": top_features,
        "probabilities": class_probabilities,
    }


def _is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _get_genomics_context() -> dict | None:
    return session.get(GENOMICS_CONTEXT_KEY)


def _get_genomics_inputs() -> dict[str, float] | None:
    return session.get(GENOMICS_INPUTS_KEY)


def _get_image_context() -> dict | None:
    return session.get(IMAGE_CONTEXT_KEY)


def _build_page_context(
    *,
    genomics_result: dict | None = None,
    image_result: dict | None = None,
    submitted_values: dict[str, float] | None = None,
    error: str | None = None,
    image_error: str | None = None,
) -> dict:
    return {
        "feature_columns": FEATURE_COLUMNS,
        "feature_meta": FEATURE_META,
        "genomics_result": genomics_result if genomics_result is not None else _get_genomics_context(),
        "image_result": image_result if image_result is not None else _get_image_context(),
        "submitted_values": submitted_values if submitted_values is not None else _get_genomics_inputs(),
        "error": error,
        "image_error": image_error,
        "genomics_metrics": model_bundle["metrics"],
        "scan_metrics": scan_model_bundle["metrics"],
        "dataset_present": DATASET_PATH.exists(),
        "scan_dataset_present": SCAN_DATASET_PATH.exists(),
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **_build_page_context())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = _parse_form_values(request.form)
        genomics_result = _predict_from_values(values)
        session[GENOMICS_INPUTS_KEY] = values
        session[GENOMICS_CONTEXT_KEY] = genomics_result
        return render_template("index.html", **_build_page_context(genomics_result=genomics_result, submitted_values=values))
    except ValueError:
        return render_template(
            "index.html",
            **_build_page_context(error="Please enter valid numeric values for every genomics field."),
        )


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    upload = request.files.get("scan_image")
    if upload is None or not upload.filename:
        return render_template(
            "index.html",
            **_build_page_context(image_error="Please choose an image file before starting the scan analysis."),
        )

    if not _is_allowed_file(upload.filename):
        return render_template(
            "index.html",
            **_build_page_context(image_error="Supported image formats are PNG, JPG, JPEG, and BMP."),
        )

    safe_name = secure_filename(upload.filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    stored_name = f"{timestamp}_{safe_name}"
    stored_path = UPLOAD_DIR / stored_name
    upload.save(stored_path)

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(stored_path) as image:
            image_result = summarize_scan_prediction(scan_model_bundle, image)
    except UnidentifiedImageError:
        stored_path.unlink(missing_ok=True)
        return render_template(
            "index.html",
            **_build_page_context(image_error="The uploaded file is not a valid image. Please try another scan file."),
        )

    image_result["image_url"] = f"uploads/{stored_name}"
    session[IMAGE_CONTEXT_KEY] = image_result
    return render_template("index.html", **_build_page_context(image_result=image_result))


@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "")
    reply = respond(
        message,
        genomics_context=_get_genomics_context(),
        image_context=_get_image_context(),
    )
    return jsonify({"reply": reply})


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "genomics_model_ready": True,
            "scan_model_ready": True,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
