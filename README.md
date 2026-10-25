# Breast Cancer Classification & FastAPI Serving Service

A modular, production-ready machine learning service built with `scikit-learn`, structured into clean pipelines, and served via `FastAPI` with `Pydantic` schema validation and `uv` dependency management.

---

## 1. Project Overview & Dataset

This project implements an end-to-end classification pipeline using the **Breast Cancer Wisconsin (Diagnostic) Dataset** from `scikit-learn`.

- **Objective:** Classify cell nuclei characteristics from digitized images of fine needle aspirates (FNA) into benign or malignant tumors.
- **Classes:**
  - `0`: **Malignant**
  - `1`: **Benign**
- **Features:** 30 continuous numeric attributes computed from digitized images (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension across mean, error, and worst values).
- **Inference Interfaces:** Dual support for batch/script inference via CLI and real-time REST API serving.

---

## 2. Project Structure

```text
ML1/
├── production_app/
│   ├── __init__.py
│   ├── api.py            # FastAPI endpoints (/health, /predict) & lifecycle handlers
│   ├── config.py         # Application settings, paths, and environment parsing
│   ├── data.py           # Dataset acquisition and train-test splitting
│   ├── evaluation.py     # Evaluation metrics, scoring, and classification reports
│   ├── model_service.py  # Model persistence (save/load) and inference handlers
│   ├── schemas.py        # Pydantic schemas for request and response validation
│   └── train.py          # Pipeline assembly and model training routines
├── run.py                # Unified CLI runner for training and batch inference
├── pyproject.toml        # Dependencies and project metadata managed by uv
├── uv.lock               # Exact dependency lockfile
├── .env.example          # Template for required environment variables
├── Dockerfile            # Container definition for containerized serving
├── .dockerignore         # Excluded files during container build
└── README.md             # Project documentation
```

---

## 3. Environment Setup & Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone [https://github.com/Ahmedzkadry/ML1.git](https://github.com/Ahmedzkadry/ML1.git)
cd ML1

# Configure environment variables
cp .env.example .env

# Install dependencies using uv
uv sync
```

---

## 4. Model Training & CLI Prediction

### Train and Save Pipeline

Run the training pipeline to fetch data, train the classifier, evaluate performance, and persist the artifact:

```bash
uv run python run.py train
```

The model artifact is exported to `artifacts/model.joblib`.

### Run CLI Prediction

Run inference on sample observations using the saved model:

```bash
uv run python run.py predict
```

---

## 5. Running the FastAPI Server

Start the API server with live reloading enabled:

```bash
uv run uvicorn production_app.api:app --host 127.0.0.1 --port 8000 --reload
```

Interactive documentation:
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 6. API Endpoints & Usage

### A. Health Check (`GET /health`)

Verifies that the service is running and the trained model pipeline is loaded in memory.

#### cURL Command:
```bash
curl -X GET [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
```

#### Response (`200 OK`):
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Response (`503 Service Unavailable`):
*(Returned if the model artifact is missing or failed to load)*
```json
{
  "status": "unhealthy",
  "model_loaded": false
}
```

---

### B. Prediction (`POST /predict`)

Accepts a single observation matching the dataset feature set and returns both numeric class and readable label.

#### cURL Command:
```bash
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.006399,
    "compactness_error": 0.04904,
    "concavity_error": 0.05373,
    "concave_points_error": 0.01587,
    "symmetry_error": 0.03003,
    "fractal_dimension_error": 0.006193,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.6,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189
  }'
```

#### Example Response (`200 OK`):
```json
{
  "predicted_class": 0,
  "readable_label": "Malignant"
}
```

#### Example Validation Error (`422 Unprocessable Entity`):
*(Returned when missing fields or invalid data types are sent)*
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": [
        "body",
        "mean_radius"
      ],
      "msg": "Field required"
    }
  ]
}
```

---

## 7. Docker Deployment

### Build the Docker Image:
```bash
docker build -t ml1-api .
```

### Run the Container:
```bash
docker run -d -p 8000:8000 --name ml_service ml1-api
```


Remember: This is an educational ML project; do not present predictions as clinically validated medical advice