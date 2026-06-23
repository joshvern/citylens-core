from .models import CitylensRequest

__all__ = ["CitylensRequest", "run_citylens"]


def __getattr__(name: str):
    # Lazily expose the pipeline entrypoint so that merely importing the
    # package — or a lightweight submodule like ``citylens_core.models`` —
    # does NOT pull the heavy stage import chain (rasterio/GDAL/SAM2). The
    # citylens-engine API only needs ``CitylensRequest`` (it triggers a Cloud
    # Run Job rather than running the pipeline in-process), so this keeps a
    # multi-second import off its cold-start path. Both
    # ``from citylens_core import run_citylens`` and ``citylens_core.run_citylens``
    # still work, resolving the real callable on first access.
    if name == "run_citylens":
        from .pipeline import run_citylens

        return run_citylens
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
