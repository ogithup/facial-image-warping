"""Face localization and cropping utilities."""


def detect_face_region(image: dict) -> dict:
    """Detect and normalize the primary face region from an image.

    Parameters
    ----------
    image:
        Preprocessed image representation.

    Returns
    -------
    dict
        Placeholder face crop, bounding box, and normalization metadata.
    """
    return {
        "face_image": {
            **image,
            "crop_applied": True,
        },
        "bounding_box": (0, 0, 0, 0),
        "detector": "placeholder",
    }
