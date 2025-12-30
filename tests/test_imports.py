def test_imports() -> None:
    from citylens_core.models import CitylensRequest
    from citylens_core.pipeline import run_citylens

    assert CitylensRequest is not None
    assert callable(run_citylens)
