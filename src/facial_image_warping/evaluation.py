"""Quantitative evaluation helpers for image transformation quality."""


def compute_mse(original_image: dict, transformed_image: dict) -> float | None:
    """Compute the Mean Squared Error between two images.

    Parameters
    ----------
    original_image:
        Reference image representation.
    transformed_image:
        Processed image representation.

    Returns
    -------
    float | None
        Placeholder metric value.
    """
    return None


def compute_psnr(original_image: dict, transformed_image: dict) -> float | None:
    """Compute the Peak Signal-to-Noise Ratio for two images."""
    return None


def compute_ssim(original_image: dict, transformed_image: dict) -> float | None:
    """Compute the Structural Similarity Index for two images."""
    return None


def evaluate_transformation(original_image: dict, transformed_image: dict) -> dict:
    """Collect the standard evaluation metrics for a transformation result.

    Parameters
    ----------
    original_image:
        Reference image representation.
    transformed_image:
        Result image representation.

    Returns
    -------
    dict
        Structured placeholder metrics for reporting and export.
    """
    return {
        "mse": compute_mse(original_image, transformed_image),
        "psnr": compute_psnr(original_image, transformed_image),
        "ssim": compute_ssim(original_image, transformed_image),
    }
