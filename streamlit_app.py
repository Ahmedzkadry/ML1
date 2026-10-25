import json
import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Diagnostic Predictor",
    page_icon="🩺",
    layout="centered",
)

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SELECTED_FEATURES_PATH = Path(
    os.getenv("SELECTED_FEATURES_PATH", "models/selected_features.json")
)

FALLBACK_FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_concavity",
    "mean_concave_points",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_concave_points",
]

# Sample realistic values for all 30 features (Benign vs Malignant)
BENIGN_SAMPLES = {
    "mean_radius": 12.05,
    "mean_texture": 14.63,
    "mean_perimeter": 78.04,
    "mean_area": 449.3,
    "mean_smoothness": 0.1031,
    "mean_compactness": 0.0909,
    "mean_concavity": 0.0659,
    "mean_concave_points": 0.0275,
    "mean_symmetry": 0.1675,
    "mean_fractal_dimension": 0.0604,
    "radius_error": 0.2636,
    "texture_error": 0.7294,
    "perimeter_error": 1.848,
    "area_error": 19.87,
    "smoothness_error": 0.0055,
    "compactness_error": 0.0143,
    "concavity_error": 0.0232,
    "concave_points_error": 0.0083,
    "symmetry_error": 0.0156,
    "fractal_dimension_error": 0.0024,
    "worst_radius": 13.76,
    "worst_texture": 20.70,
    "worst_perimeter": 89.88,
    "worst_area": 582.6,
    "worst_smoothness": 0.1494,
    "worst_compactness": 0.2156,
    "worst_concavity": 0.3050,
    "worst_concave_points": 0.0655,
    "worst_symmetry": 0.2747,
    "worst_fractal_dimension": 0.0830,
}

MALIGNANT_SAMPLES = {
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.80,
    "mean_area": 1001.0,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.0787,
    "radius_error": 1.095,
    "texture_error": 0.9053,
    "perimeter_error": 8.589,
    "area_error": 153.4,
    "smoothness_error": 0.0064,
    "compactness_error": 0.0490,
    "concavity_error": 0.0537,
    "concave_points_error": 0.0159,
    "symmetry_error": 0.0300,
    "fractal_dimension_error": 0.0062,
    "worst_radius": 25.38,
    "worst_texture": 17.33,
    "worst_perimeter": 184.60,
    "worst_area": 2019.0,
    "worst_smoothness": 0.1622,
    "worst_compactness": 0.6656,
    "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654,
    "worst_symmetry": 0.4601,
    "worst_fractal_dimension": 0.1189,
}


def get_selected_features() -> list[str]:
    if SELECTED_FEATURES_PATH.exists():
        try:
            data = json.loads(SELECTED_FEATURES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [str(f) for f in data]
        except Exception:
            pass
    return FALLBACK_FEATURES


st.title("🩺 Breast Cancer Diagnostic Classifier")
st.markdown(
    "Enter the **10 selected diagnostic features** below and click **Predict** "
    "to query the FastAPI inference service over HTTP."
)

api_url = st.sidebar.text_input("FastAPI Base URL", value=DEFAULT_API_URL).rstrip("/")

if st.sidebar.button("Check API Health"):
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        if resp.status_code == 200:
            st.sidebar.success(f"API Healthy: {resp.json()}")
        else:
            st.sidebar.warning(f"API Status {resp.status_code}: {resp.text}")
    except requests.RequestException as exc:
        st.sidebar.error(f"Cannot reach API: {exc}")

selected_features = get_selected_features()

sample_choice = st.radio(
    "Pre-fill sample values:",
    options=["Benign Sample", "Malignant Sample"],
    horizontal=True,
)
active_defaults = (
    BENIGN_SAMPLES if sample_choice == "Benign Sample" else MALIGNANT_SAMPLES
)

with st.form("prediction_form"):
    st.subheader(f"Input Features ({len(selected_features)} Selected Features)")
    cols = st.columns(2)
    payload: dict[str, float] = {}

    for idx, feat in enumerate(selected_features):
        col = cols[idx % 2]
        default_val = float(active_defaults.get(feat, 1.0))
        payload[feat] = col.number_input(
            label=feat.replace("_", " ").title(),
            min_value=0.0,
            value=default_val,
            step=0.01,
            format="%.4f",
            key=f"{sample_choice}_{feat}",
        )

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    try:
        response = requests.post(
            f"{api_url}/predict",
            json=payload,
            timeout=10,
        )
        if response.status_code == 200:
            result = response.json()
            label = result.get("readable_label", "Unknown")
            pred_class = result.get("predicted_class", -1)
            if pred_class == 1:
                st.success(f"**Prediction:** {label} (Class `{pred_class}`)")
            else:
                st.error(f"**Prediction:** {label} (Class `{pred_class}`)")
        elif response.status_code == 422:
            st.warning(f"Validation Error (422): {response.json()}")
        elif response.status_code == 503:
            st.error("Model Service Unavailable (503): Model is not loaded on server.")
        else:
            st.error(f"Unexpected Error ({response.status_code}): {response.text}")
    except requests.Timeout:
        st.error(
            "Request timed out. Please verify that the FastAPI server is responsive."
        )
    except requests.ConnectionError:
        st.error(
            f"Could not connect to FastAPI at `{api_url}`. "
            "Make sure Uvicorn or the Docker container is running."
        )
    except requests.RequestException as exc:
        st.error(f"HTTP Request failed: {exc}")
