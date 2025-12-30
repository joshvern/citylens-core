from pathlib import Path

import json


def test_pipeline_smoke(tmp_path: Path) -> None:
    from citylens_core import CitylensRequest, run_citylens

    req = CitylensRequest(address="test", segmentation_backend="unet")
    artifacts = run_citylens(req, tmp_path)

    assert set(artifacts.keys()) == {"preview", "change", "mesh", "summary"}

    assert artifacts["preview"].exists()
    assert artifacts["change"].exists()
    assert artifacts["mesh"].exists()
    assert artifacts["summary"].exists()

    payload = json.loads(artifacts["summary"].read_text())
    assert "warnings" in payload
    assert "errors" in payload


def test_outputs_gating_still_writes_contract(tmp_path: Path) -> None:
    from citylens_core import CitylensRequest, run_citylens

    # Disable preview + mesh stages; contract files must still exist (placeholders allowed).
    req = CitylensRequest(address="test", segmentation_backend="unet", outputs=["change"])
    artifacts = run_citylens(req, tmp_path)

    assert set(artifacts.keys()) == {"preview", "change", "mesh", "summary"}
    for p in artifacts.values():
        assert p.exists()
