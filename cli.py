"""
CLI (Command Line Interface) for the Nature Scene Classifier.

This script allows users to classify an image directly from the terminal
"""

import argparse
from pathlib import Path
from fastai.vision.all import PILImage
from core.predict import predict_image_pil
from typing import Dict


def classify_image(image_path: Path) -> Dict[str, float]:
    """
    Load an image from disk and return prediction probabilities.

    Parameters
    ----------
    image_path : Path
        Path to the image file.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping class names to probabilities.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        img = PILImage.create(image_path)
    except Exception as e:
        raise RuntimeError(f"Could not open image: {e}")

    return predict_image_pil(img)


def main():
    """
    Entry point for CLI execution.
    Uses argparse to parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Classify a nature image into one of the following categories: "
            "forest, beach, bird, fish."
        )
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image file to classify."
    )

    args = parser.parse_args()
    image_path = Path(args.image_path)

    try:
        results = classify_image(image_path)
    except Exception as e:
        print(f"\nError: {e}\n")
        return

    print("\n🌿 Nature Scene Classifier — Prediction Results:\n")
    for label, prob in results.items():
        print(f"  {label:<10}: {prob:.4f}")
    print("")


if __name__ == "__main__":
    main()
