from __future__ import annotations

from typing import Any


DISCLAIMER = (
    "This chatbot is educational and supports the demo model only. "
    "It should not be used as a medical diagnosis."
)


FEATURE_HELP = {
    "tp53_expression": "TP53 expression is a tumor suppressor marker often linked with DNA-damage response.",
    "brca1_expression": "BRCA1 expression is commonly discussed in hereditary breast and ovarian cancer pathways.",
    "egfr_expression": "EGFR expression can be relevant in several epithelial cancers, especially lung cancer biology.",
    "kras_expression": "KRAS expression is useful when discussing oncogenic signaling, especially in colorectal and lung disease.",
    "pik3ca_expression": "PIK3CA expression is associated with PI3K pathway activation and can matter in breast-like tumors.",
    "tumor_mutational_burden": "Tumor mutational burden is a genomic summary of how many mutations are present in the tumor.",
    "msi_score": "MSI score reflects microsatellite instability and is important in colorectal-type genomic patterns.",
    "copy_number_instability": "Copy number instability measures large-scale genomic gains and losses.",
    "patient_age": "Patient age is included as context for the demo model, not as a stand-alone diagnostic marker.",
    "smoking_index": "Smoking index approximates tobacco exposure and is often relevant in lung cancer risk discussions.",
}


def prediction_summary(prediction: str, confidence: float, top_features: list[str]) -> str:
    features_text = ", ".join(top_features[:3]) if top_features else "the submitted genomic markers"
    return (
        f"The model's highest-probability class is {prediction} "
        f"with {confidence:.1%} confidence. The strongest contributing markers in this demo are "
        f"{features_text}. {DISCLAIMER}"
    )


def image_summary(prediction: str, confidence: float, signal_lines: list[str], explanation: str) -> str:
    signal_text = ", ".join(signal_lines[:3]) if signal_lines else "the scan texture and brightness signals"
    return (
        f"The latest uploaded scan was classified as {prediction} with {confidence:.1%} confidence. "
        f"Key image signals: {signal_text}. {explanation} {DISCLAIMER}"
    )


def respond(
    message: str,
    genomics_context: dict[str, Any] | None = None,
    image_context: dict[str, Any] | None = None,
) -> str:
    text = message.lower().strip()

    if not text:
        return "Please type a question about the genomics model, biomarkers, uploaded scan, or the latest prediction."

    if any(word in text for word in ["hello", "hi", "hey"]):
        return (
            "Hello. I can explain the genomics features, summarize the latest prediction, "
            "interpret the latest uploaded scan, or suggest what to review next. "
            + DISCLAIMER
        )

    if "disclaimer" in text or "safe" in text or "medical advice" in text:
        return DISCLAIMER

    if "latest prediction" in text or "last prediction" in text or "result" in text:
        if "image" in text or "xray" in text or "x-ray" in text or "scan" in text:
            if image_context:
                return image_summary(
                    prediction=image_context["predicted_class"],
                    confidence=image_context["confidence"],
                    signal_lines=image_context["signal_lines"],
                    explanation=image_context["explanation"],
                )
            return "There is no uploaded scan result in the current session yet."

        if genomics_context:
            return prediction_summary(
                prediction=genomics_context["predicted_class"],
                confidence=genomics_context["confidence"],
                top_features=genomics_context["top_features"],
            )
        if image_context:
            return image_summary(
                prediction=image_context["predicted_class"],
                confidence=image_context["confidence"],
                signal_lines=image_context["signal_lines"],
                explanation=image_context["explanation"],
            )
        return "There is no prediction in the current session yet. Submit genomic values or upload a scan first."

    if "feature" in text or "marker" in text or "gene" in text:
        for feature_name, explanation in FEATURE_HELP.items():
            readable = feature_name.replace("_", " ")
            if feature_name in text or readable in text:
                return explanation
        return (
            "This demo uses TP53, BRCA1, EGFR, KRAS, PIK3CA, tumor mutational burden, MSI score, "
            "copy number instability, patient age, and smoking index."
        )

    if "breast" in text:
        return (
            "A breast-like result in this demo tends to align with higher BRCA1 and PIK3CA activity "
            "plus moderate copy number instability."
        )

    if "lung" in text:
        return (
            "A lung-like result in this demo is usually driven by higher EGFR, KRAS, tumor mutational burden, "
            "and smoking exposure."
        )

    if "colorectal" in text or "colon" in text:
        return (
            "A colorectal-like result in this demo is strongly associated with higher KRAS and MSI-related scores."
        )

    if "next step" in text or "what should i do" in text:
        return (
            "For this demo, the best next step is to review the class probabilities, inspect the strongest markers "
            "or scan signals, and compare them with clinically validated findings before making any real-world conclusion."
        )

    if "image" in text or "scan" in text or "xray" in text or "x-ray" in text:
        if image_context and ("latest" in text or "upload" in text or "summary" in text):
            return image_summary(
                prediction=image_context["predicted_class"],
                confidence=image_context["confidence"],
                signal_lines=image_context["signal_lines"],
                explanation=image_context["explanation"],
            )
        return (
            "The image module reviews an uploaded scan, predicts a demo scan pattern, and explains the result using "
            "brightness, asymmetry, hotspot activity, and edge-related signals."
        )

    if "chatbot" in text or "help" in text:
        return (
            "I can explain genomics biomarkers, summarize the latest genomic prediction, interpret the latest uploaded "
            "scan result, or describe what the cancer-pattern classes mean in this demo."
        )

    return (
        "I can help explain biomarkers, summarize the latest genomics prediction, interpret the latest uploaded scan, "
        "or describe what each cancer-pattern class means in this demo project."
    )
