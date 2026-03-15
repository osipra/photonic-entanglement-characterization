# Photonic Entanglement Characterization

A Python toolkit for **photonic entangled-state characterization**, built from real quantum optics lab workflows and reorganized into a cleaner, reusable package.

## Purpose

This repository is designed to turn analysis that would normally remain buried inside lab notebooks into a more structured software project.

Its main goals are to support workflows such as:

- quantum state tomography
- Bell-state characterization
- CHSH / Bell-inequality analysis
- waveplate and analyzer-setting utilities
- later, photonic interference and HOM-style analysis

This is not intended to be a generic quantum computing tutorial repo. The focus is on **lab-oriented photonic QIP analysis**.

## Current structure

```text
src/pec/
tests/
notebooks/
docs/