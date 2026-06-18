import inspect

import fusion


def test_top_level_callables_present() -> None:
    for name in (
        "load_config",
        "load_data",
        "prepare",
        "sample",
        "project",
        "run",
        "save_weights",
        "save_metadata",
        "plot_projection",
    ):
        assert hasattr(fusion, name), f"fusion.{name} missing"
        assert callable(getattr(fusion, name))


def test_run_signature_takes_config() -> None:
    sig = inspect.signature(fusion.run)
    assert list(sig.parameters) == ["cfg"]
