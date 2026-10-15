import sys

from production_app.data import load_raw_data
from production_app.main import run_prediction_pipeline, run_training_pipeline

mode = sys.argv[1] if len(sys.argv) > 1 else "train"

if mode == "train":
    metrics = run_training_pipeline()
    print(f"Train Accuracy: {metrics['accuracy']:.4f}")
elif mode == "predict":
    sample = load_raw_data().drop(columns=["target"]).head(5)
    print(f"Predictions: {run_prediction_pipeline(sample)}")
