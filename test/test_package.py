"""Package-level smoke tests."""

import heft


def test_heft_package_is_importable() -> None:
    assert heft.__name__ == "heft"
