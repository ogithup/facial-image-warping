"""Quantitative evaluation helpers for image transformation quality."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity


EVALUATION_OUTPUT_DIR = Path("outputs/evaluation")


def _to_uint8_pixels(image: dict) -> tuple[np.ndarray, str]:
    """Convert supported image payloads to uint8 pixel data."""
    pixels = image["pixels"]
    if pixels.dtype.kind == "f":
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
    return pixels, image.get("color_space", "BGR")


def _prepare_pair(original_image: dict, transformed_image: dict) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize two images into same-shaped RGB uint8 arrays."""
    original_pixels, original_color_space = _to_uint8_pixels(original_image)
    transformed_pixels, transformed_color_space = _to_uint8_pixels(transformed_image)

    if original_pixels.shape != transformed_pixels.shape:
        raise ValueError(
            "Original and transformed images must share the same shape for evaluation. "
            f"Got {original_pixels.shape} and {transformed_pixels.shape}."
        )

    def to_rgb(pixels: np.ndarray, color_space: str) -> np.ndarray:
        if color_space == "RGB":
            return pixels
        if color_space == "BGR":
            return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
        if color_space == "GRAYSCALE":
            return cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
        raise ValueError(f"Unsupported color space for evaluation: {color_space}")

    return to_rgb(original_pixels, original_color_space), to_rgb(transformed_pixels, transformed_color_space)


def compute_mse(original_image: dict, transformed_image: dict) -> float:
    """Compute Mean Squared Error between two aligned images."""
    original_rgb, transformed_rgb = _prepare_pair(original_image, transformed_image)
    difference = original_rgb.astype(np.float32) - transformed_rgb.astype(np.float32)
    return float(np.mean(difference ** 2))


def compute_psnr(original_image: dict, transformed_image: dict) -> float:
    """Compute Peak Signal-to-Noise Ratio from the MSE value."""
    mse = compute_mse(original_image, transformed_image)
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(original_image: dict, transformed_image: dict) -> float:
    """Compute Structural Similarity Index between two images."""
    original_rgb, transformed_rgb = _prepare_pair(original_image, transformed_image)
    return float(
        structural_similarity(
            original_rgb,
            transformed_rgb,
            channel_axis=-1,
            data_range=255,
        )
    )


def create_image_difference_visualization(
    original_image: dict,
    transformed_image: dict,
    output_path: str | Path,
) -> Path:
    """Create a visualization showing original, transformed, and absolute difference images."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    original_rgb, transformed_rgb = _prepare_pair(original_image, transformed_image)
    absolute_difference = cv2.absdiff(original_rgb, transformed_rgb)
    grayscale_difference = cv2.cvtColor(absolute_difference, cv2.COLOR_RGB2GRAY)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(transformed_rgb)
    plt.title("Transformed")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(grayscale_difference, cmap="inferno")
    plt.title("Absolute Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file


def export_evaluation_to_csv(results: dict, output_path: str | Path) -> Path:
    """Export evaluation metrics to a CSV file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mse",
                "psnr",
                "ssim",
                "mean_absolute_difference",
                "max_absolute_difference",
            ],
        )
        writer.writeheader()
        writer.writerow({key: results[key] for key in writer.fieldnames})
    return output_file


def evaluate_transformation(
    original_image: dict,
    transformed_image: dict,
    save_outputs: bool = True,
) -> dict:
    """Collect evaluation metrics and optional difference artifacts for reporting."""
    original_rgb, transformed_rgb = _prepare_pair(original_image, transformed_image)
    absolute_difference = cv2.absdiff(original_rgb, transformed_rgb)
    mean_absolute_difference = float(np.mean(absolute_difference))
    max_absolute_difference = int(np.max(absolute_difference))

    stem = Path(transformed_image.get("file_name", original_image.get("file_name", "image.png"))).stem
    difference_path = EVALUATION_OUTPUT_DIR / f"{stem}_difference.png"
    csv_path = EVALUATION_OUTPUT_DIR / f"{stem}_metrics.csv"

    results = {
        "mse": compute_mse(original_image, transformed_image),
        "psnr": compute_psnr(original_image, transformed_image),
        "ssim": compute_ssim(original_image, transformed_image),
        "mean_absolute_difference": mean_absolute_difference,
        "max_absolute_difference": max_absolute_difference,
        "difference_path": str(difference_path),
        "csv_path": str(csv_path),
    }

    if save_outputs:
        create_image_difference_visualization(original_image, transformed_image, difference_path)
        export_evaluation_to_csv(results, csv_path)

    return results
