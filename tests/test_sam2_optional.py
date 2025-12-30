import json
import importlib.util
from pathlib import Path

import pytest


def test_sam2_optional(tmp_path: Path) -> None:
    sam2_spec = importlib.util.find_spec("sam2")
    if sam2_spec is None:
        pytest.skip("sam2 not installed")

    from citylens_core import CitylensRequest, run_citylens

    # Intentionally point at default asset locations that may not exist.
    req = CitylensRequest(address="test", segmentation_backend="sam2")
    artifacts = run_citylens(req, tmp_path)

    assert artifacts["preview"].exists()
    assert artifacts["change"].exists()
    assert artifacts["mesh"].exists()
    assert artifacts["summary"].exists()

    payload = json.loads(artifacts["summary"].read_text())
    # Should not crash even if weights missing; should warn instead.
    assert isinstance(payload.get("warnings"), list)
