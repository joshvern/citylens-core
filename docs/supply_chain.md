# Supply chain and release evidence

`pyproject.toml` is the published package contract. `uv.lock` is the
authoritative contributor and CI resolution and must remain synchronized with
that contract.

## Local verification

Run the same lightweight development and LiDAR graph used by CI:

```bash
uv lock --check
uv sync --extra dev --extra lidar --frozen
./.venv/bin/python -m ruff check src tests
./.venv/bin/python -m pytest
```

The SAM2 extra is not installed by the core CI job because the generic PyPI
Torch distribution may include multi-gigabyte accelerator packages. CI still
exports and audits the complete published SAM2/LiDAR dependency graph and
includes it in the CycloneDX SBOM. `citylens-engine` is the authoritative
integration and container test for the production CPU-only Torch graph.

## CI release evidence

Every pull request:

1. rejects a stale lockfile;
2. runs Ruff and the full unit suite on Python 3.11;
3. audits the complete locked runtime, LiDAR, and SAM2 graph with
   `pip-audit`;
4. emits a CycloneDX 1.5 SBOM;
5. builds the sdist and wheel without workspace-only sources; and
6. validates both distributions with Twine.

The SBOM and distributions are retained as the `core-release-evidence`
workflow artifact for 30 days. GitHub Actions are commit-pinned and Dependabot
tracks both the uv lock and workflow actions.

## Dependency policy

- The SAM2 source dependency is pinned to an immutable upstream commit.
- Security-fix floors in `pyproject.toml` protect package consumers even when
  they do not use this repository's lockfile.
- A dependency update is accepted only when the lock, audit, tests, package
  build, and downstream engine CPU integration are all green.
- Production engine images pin `citylens-core` by immutable commit and carry
  their own SBOM and container vulnerability evidence.
