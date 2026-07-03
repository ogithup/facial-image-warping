"""Input acquisition utilities for facial image processing workflows."""


def validate_image_source(image_source: str) -> bool:
    """Validate whether the provided image source is acceptable.

    Parameters
    ----------
    image_source:
        Path, filename, or external identifier for an input image.

    Returns
    -------
    bool
        Placeholder validation result. Replace with file existence, extension,
        and resolution checks in later sprints.
    """
    return bool(image_source)


def request_image_input(image_source: str) -> dict:
    """Load or register a user-provided image for the DSP pipeline.

    Parameters
    ----------
    image_source:
        Path or identifier for a user-selected facial image.

    Returns
    -------
    dict
        Placeholder container for the image payload and input metadata.
    """
    if not validate_image_source(image_source):
        raise ValueError("Invalid image source provided.")

    return {
        "source": image_source,
        "image": {
            "path": image_source,
            "pixels": None,
            "color_space": "unknown",
        },
    }
