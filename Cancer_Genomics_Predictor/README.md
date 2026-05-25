# Cancer Genomics Predictor with Scan Analysis and Chatbot

This project is a standalone machine learning demo for cancer genomics classification with a companion chatbot and uploaded scan analysis. It predicts a likely cancer pattern from genomics-inspired markers, analyzes an uploaded scan image, and explains the result in plain language.

## What the project includes

- A synthetic genomics-style dataset generator for offline development
- A `RandomForestClassifier` model for multiclass cancer pattern prediction
- A synthetic scan-pattern dataset generator built from lightweight image features
- A Flask web app with a genomics input form, scan upload section, and probability dashboards
- A local chatbot that explains biomarkers, uploaded scan results, class meanings, and the latest prediction

## Predicted classes

- `Breast-like`
- `Lung-like`
- `Colorectal-like`

## Uploaded scan classes

- `Normal-like Scan`
- `Localized Nodule-like Pattern`
- `Mass-like Pattern`

## Genomics-style input features

- `tp53_expression`
- `brca1_expression`
- `egfr_expression`
- `kras_expression`
- `pik3ca_expression`
- `tumor_mutational_burden`
- `msi_score`
- `copy_number_instability`
- `patient_age`
- `smoking_index`

## Project structure

```text
Cancer_Genomics_Predictor/
|- app.py
|- chatbot.py
|- data_utils.py
|- image_model.py
|- train_model.py
|- requirements.txt
|- README.md
|- artifacts/
|- data/
|- static/
`- templates/
```

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model manually if you want to pre-generate artifacts:

```bash
python train_model.py
```

3. The scan model is trained automatically on first app run, or you can trigger it from Python by importing `image_model.py`.

4. Start the Flask app:

```bash
python app.py
```

5. Open the local URL shown by Flask in your browser.

## Deploy on Render

This repository now includes a repo-root `render.yaml` and a project-level `Procfile`, so the app is ready for a Render web service deployment.

### Deployment steps

1. Push this repository to GitHub.
2. Create a new Render Web Service or Deploy Blueprint from the connected repository.
3. Render should use:
   - Root directory: `ML-CaPsule/Cancer_Genomics_Predictor`
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn wsgi:app --bind 0.0.0.0:$PORT`
4. Set the `SECRET_KEY` environment variable if you are not using the generated one from `render.yaml`.
5. Deploy and open the generated public URL.

### Deployment notes

- The chatbot is shown directly on the website homepage and works through the `/chat` endpoint.
- User prediction context is stored per session so different visitors do not overwrite each other's latest results.
- Uploaded scan previews are stored in `static/uploads/`, which is suitable for demo hosting but not long-term medical storage.
- A health check endpoint is available at `GET /health`.

## API routes

- `GET /` renders the web application
- `POST /predict` predicts the most likely cancer pattern
- `POST /analyze-image` analyzes an uploaded scan image
- `POST /chat` returns a chatbot response in JSON
- `GET /health` returns a deployment health response

Example chatbot payload:

```json
{
  "message": "Explain the MSI score"
}
```

## Important note

This project is an educational demo built with synthetic genomics-inspired data and synthetic scan-pattern training data so it can run offline inside this repository. It is not a clinical diagnostic tool and should not be used for medical decisions.
