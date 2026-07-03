"""Frequency-domain DSP analysis helpers for facial image processing."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


FREQUENCY_OUTPUT_DIR = Path("outputs/frequency")


def _to_grayscale_float(image: dict) -> np.ndarray:
    """Convert supported image payloads into grayscale float32 for FFT."""
    pixels = image["pixels"]
    if pixels.dtype.kind == "f":
        pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)

    color_space = image.get("color_space", "BGR")
    if color_space == "GRAYSCALE":
        grayscale = pixels
    elif color_space == "BGR":
        grayscale = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    elif color_space == "RGB":
        grayscale = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported color space for Fourier analysis: {color_space}")

    return grayscale.astype(np.float32)


def compute_fft(image: dict) -> np.ndarray:
    """Compute the centered 2D Fast Fourier Transform of an image."""
    grayscale = _to_grayscale_float(image)
    fft_result = np.fft.fft2(grayscale)
    return np.fft.fftshift(fft_result)


def compute_magnitude_spectrum(fft_result: np.ndarray) -> np.ndarray:
    """Compute a log-scaled magnitude spectrum from a centered FFT result."""
    magnitude = np.abs(fft_result)
    return np.log1p(magnitude)


def calculate_frequency_energy(fft_result: np.ndarray) -> dict:
    """Calculate total, low-frequency, and high-frequency spectral energy."""
    power_spectrum = np.abs(fft_result) ** 2
    height, width = power_spectrum.shape
    center_y, center_x = height // 2, width // 2
    radius = max(1, int(min(height, width) * 0.1))

    y_indices, x_indices = np.ogrid[:height, :width]
    mask = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2 <= radius ** 2

    total_energy = float(power_spectrum.sum())
    low_frequency_energy = float(power_spectrum[mask].sum())
    high_frequency_energy = float(power_spectrum[~mask].sum())
    high_low_ratio = float(high_frequency_energy / low_frequency_energy) if low_frequency_energy > 0 else float("inf")

    return {
        "total_energy": total_energy,
        "low_frequency_energy": low_frequency_energy,
        "high_frequency_energy": high_frequency_energy,
        "high_low_ratio": high_low_ratio,
        "low_frequency_radius": radius,
    }


def visualize_spectrum(image: dict, spectrum: np.ndarray, output_path: str | Path) -> Path:
    """Visualize original grayscale image and magnitude spectrum side by side."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    grayscale = _to_grayscale_float(image)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(grayscale, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(spectrum, cmap="magma")
    plt.title("Log Magnitude Spectrum")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file


def export_frequency_analysis_to_csv(results: dict, output_path: str | Path) -> Path:
    """Export frequency energy metrics to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "total_energy",
                "low_frequency_energy",
                "high_frequency_energy",
                "high_low_ratio",
                "low_frequency_radius",
            ],
        )
        writer.writeheader()
        writer.writerow(results)
    return output_file


def analyze_frequency_content(image: dict, save_outputs: bool = True) -> dict:
    """Run the full Fourier analysis workflow for an image."""
    fft_result = compute_fft(image)
    magnitude_spectrum = compute_magnitude_spectrum(fft_result)
    energy_metrics = calculate_frequency_energy(fft_result)

    stem = Path(image.get("file_name", "image.png")).stem
    spectrum_path = FREQUENCY_OUTPUT_DIR / f"{stem}_spectrum.png"
    csv_path = FREQUENCY_OUTPUT_DIR / f"{stem}_frequency_metrics.csv"
    if save_outputs:
        visualize_spectrum(image, magnitude_spectrum, spectrum_path)
        export_frequency_analysis_to_csv(energy_metrics, csv_path)

    return {
        "fft": fft_result,
        "magnitude_spectrum": magnitude_spectrum,
        **energy_metrics,
        "image_reference": image,
        "spectrum_path": str(spectrum_path),
        "csv_path": str(csv_path),
    }

