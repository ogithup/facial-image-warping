"""Preprocessing stage for normalization and analysis readiness."""


def resize_to_standard(image: dict, target_size: tuple[int, int] = (512, 512)) -> dict:
    """Resize the input image to a standard resolution.

    Parameters
    ----------
    image:
        Placeholder image object or metadata dictionary.
    target_size:
        Expected width and height for consistent downstream evaluation.

    Returns
    -------
    dict
        Placeholder resized image representation.
    """
    return {
        **image,
        "target_size": target_size,
    }


def convert_to_grayscale(image: dict) -> dict:
    """Convert an image to grayscale when frequency analysis requires it.

    Parameters
    ----------
    image:
        Placeholder image object or metadata dictionary.

    Returns
    -------
    dict
        Placeholder grayscale image representation.
    """
    return {
        **image,
        "color_space": "grayscale",
    }


def preprocess_image(image: dict) -> dict:
    """Apply the standard preprocessing sequence for the pipeline.

    Parameters
    ----------
    image:
        Raw image payload from the input module.

    Returns
    -------
    dict
        Structured preprocessing output for later modules.
    """
    resized = resize_to_standard(image)
    grayscale = convert_to_grayscale(resized)
    return {
        "image": grayscale,
        "status": "preprocessed",
    }
