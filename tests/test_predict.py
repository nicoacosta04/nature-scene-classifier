import sys
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from core.predict import predict_image_pil
from fastai.vision.all import PILImage


def test_prediction_output():
    sample_path = ROOT / "assets" / "sample.jpg"
    img = PILImage.create(sample_path)

    results = predict_image_pil(img)

    assert isinstance(results, dict)
    assert len(results) > 0

    for label, prob in results.items():
        assert isinstance(label, str)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0