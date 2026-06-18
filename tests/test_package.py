from importlib.metadata import version

import fusion


def test_version_matches_distribution_metadata() -> None:
    assert fusion.__version__ == version("ice-fusion")
