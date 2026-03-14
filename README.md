# Photonic Entanglement Characterization

Scaffold for a photonic quantum information analysis toolkit focused on lab-facing workflows such as tomography, Bell/CHSH analysis, waveplate calculations, metrics, IO, and plotting.

## Project layout

- `src/pec/`: package source code
- `tests/`: package smoke tests and public API checks
- `notebooks/`: home for future analysis notebooks

The current package is intentionally a scaffold only. It defines module structure, public imports, docstrings, and function signatures without implementing the underlying physics yet.

## Quick start

```bash
pip install -e .[dev]
pytest
```

