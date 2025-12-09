"""
Core prediction module for the Nature Scene Classifier.

This module loads the exported FastAI model and exposes a clean,
production-ready function for performing inference on PIL images.

The function predict_image_pil() is intentionally lightweight so it can
be reused by:
    - Gradio UI (app.py)
    - CLI tool (cli.py)
    - Unit tests (tests/test_predict.py)
    - FastAPI service (future extension)
"""

from fastai.vision.all import *
from pathlib import Path
from typing import Dict, Any


# --------------------------------------------------------------------------------------
# Model loading (done once at import)
# --------------------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent.parent / "models" / "nature_scene_classifier_wsl_v2.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model file not found at: {}".format(MODEL_PATH))

try:
    learn = load_learner(MODEL_PATH)
except Exception as e:
    raise RuntimeError("Failed to load model: {}".format(e))

labels = learn.dls.vocab


# --------------------------------------------------------------------------------------
# Prediction function
# --------------------------------------------------------------------------------------

def predict_image_pil(img: PILImage) -> Dict[str, float]:
    """
    Predict probabilities for each class given a PIL image.

    Parameters
    ----------
    img : PILImage
        A PIL image object.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping class names to probabilities between 0.0 and 1.0.

    Raises
    ------
    ValueError
        If the input image is None or invalid.
    RuntimeError
        If prediction fails.
    """
    if img is None:
        raise ValueError("Input image is None.")

    if not isinstance(img, (PILImage, Image.Image)):
        raise ValueError("Expected PIL image, got {}".format(type(img)))

    try:
        pred, pred_idx, probs = learn.predict(img)
        return {labels[i]: float(probs[i]) for i in range(len(labels))}
    except Exception as e:
        raise RuntimeError("Prediction failed: {}".format(e))