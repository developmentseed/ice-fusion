import fusion


def test_version_exposed():
    assert isinstance(fusion.__version__, str)
    assert fusion.__version__
