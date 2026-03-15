# AGENTS.md

This repository is a photonic quantum information analysis toolkit focused on entangled-state characterization.

## Project intent
The repository is meant to be a clean, public-facing software project derived from photonic QIP lab workflows.

It is not a storage location for raw course deliverables.
It should prioritize reusable code, tests, documentation, and polished demos.

## Rules
- Core package uses Python + NumPy + SciPy + matplotlib + pandas.
- Do not make Qiskit a required dependency.
- QuTiP is optional only and should not be required for the core package.
- Main package code must not depend on notebook cell state.
- Preserve a lab-analysis focus: tomography, Bell/CHSH, waveplates, metrics, IO, plotting.
- Keep functions small, typed where practical, and documented.
- Add tests for reusable functions.
- Prefer explicit, readable physics over clever abstractions.
- Avoid duplicating logic across modules.
- Keep the public API clean and lab-oriented.

## Repository structure expectations
- `src/pec/` contains the reusable package code.
- `tests/` contains unit tests.
- `notebooks/` contains polished demos and usage examples only.
- `docs/` contains project documentation and conceptual guides.
- Raw lab notebooks, draft coursework, and messy intermediate files should not be added to the public repo.

## Notebook policy
- Do not treat raw notebooks as permanent public artifacts.
- Only polished demonstration notebooks should remain in the repo.
- Do not rewrite notebooks unless explicitly asked.
- When notebook logic is reusable, extract it into `src/pec/` instead of leaving it embedded in notebook cells.

## Scope guidance
- `states.py`: standard states, Bell states, projectors, tensor helpers, operator cleanup helpers
- `metrics.py`: purity, fidelities, Bell-state fidelities, related state metrics
- `bell.py`: Bell-analysis helpers tied to photonic polarization experiments
- `chsh.py`: CHSH-specific evaluation logic
- `mle.py`: reusable MLE and parameterization utilities
- `tomography.py`: high-level state reconstruction workflows
- `waveplates.py`: analyzer and waveplate-setting utilities
- `io.py`: lab-facing data ingestion and formatting helpers
- `plotting.py`: reusable plotting helpers for matrices, Bell/CHSH, tomography, and later interference

## Public-facing standard
Changes should make the repository look more like a professional scientific software project and less like a course submission archive.