"""Basic smoke tests for the project scaffold."""

from app import run_pipeline


def test_pipeline_summary_contains_expected_keys() -> None:
    """Verify the scaffolded pipeline returns the planned output structure."""
    result = run_pipeline("sample_face.png")

    expected_keys = {
        "original_image",
        "preprocessed_image",
        "face_image",
        "landmarks",
        "warped_image",
        "aged_image",
        "deaged_image",
        "original_frequency",
        "transformed_frequency",
        "metrics",
    }

    assert expected_keys.issubset(result.keys())
