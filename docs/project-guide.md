\# Project Guide



\## Purpose



This repository is a reusable Python toolkit for photonic entangled-state characterization. Its goal is to turn the kinds of calculations that usually live inside lab notebooks into a cleaner, more reusable package.



The long-term focus of the project is:



\- quantum state tomography

\- Bell-state characterization

\- CHSH / Bell-inequality analysis

\- waveplate-setting utilities

\- later, interference and HOM-style analysis



This repo is not meant to be a generic quantum-computing tutorial. It is meant to reflect the kinds of workflows that are actually useful in a photonic QIP lab.



\## Design Philosophy



This project follows a few simple principles:



\- Scientific Python first

\- Lab-analysis oriented

\- Small reusable functions

\- Notebooks for demonstrations, package for logic

\- Readable physics over clever abstraction



\## Repository Structure



src/pec/

\- states.py

\- metrics.py

\- mle.py

\- tomography.py

\- bell.py

\- chsh.py

\- waveplates.py

\- io.py

\- plotting.py



tests/

notebooks/

docs/



The package root is `src/pec`.



\## Module Map



\### `states.py`

Defines the basic quantum objects used throughout the project.



This includes:

\- computational basis states

\- polarization basis states

\- Bell states

\- projector helpers

\- tensor-product helpers

\- Pauli matrices

\- Hermitian / trace-normalization utilities



This is the base layer of the project.



\### `metrics.py`

Defines small reusable quantities that can be computed from states.



This includes:

\- purity

\- pure-state fidelity

\- density-matrix fidelity

\- Bell-state fidelities

\- trace distance

\- linear entropy



This module sits on top of `states.py`.



\### `bell.py`

Will hold Bell-analysis helpers.



Planned contents include:

\- two-qubit correlators

\- expectation-value helpers

\- Bell-analysis utilities tied directly to polarization experiments



\### `chsh.py`

Will hold CHSH-specific routines.



Planned contents include:

\- CHSH parameter evaluation

\- optimization of analyzer settings

\- utilities for comparing ideal and reconstructed states



\### `tomography.py`

Will hold state-reconstruction workflows.



Planned contents include:

\- one-qubit tomography

\- two-qubit tomography

\- measurement-operator assembly

\- high-level reconstruction helpers



\### `mle.py`

Will hold optimization and parameterization routines used by tomography.



Planned contents include:

\- physical density-matrix parameterizations

\- negative log-likelihood functions

\- constrained fitting helpers



\### `waveplates.py`

Will hold waveplate and analyzer-setting utilities.



Planned contents include:

\- QWP/HWP Jones matrices

\- settings-to-state conversions

\- fitting or solving for analyzer settings from target measurement directions



\### `io.py`

Will hold data-loading and parsing helpers.



\### `plotting.py`

Will hold plotting helpers for results and diagnostics.



\## Reading Order



A new reader should approach the project in this order:



1\. `states.py`

2\. `metrics.py`

3\. `bell.py`

4\. `chsh.py`

5\. `tomography.py`

6\. `mle.py`

7\. `waveplates.py`



\## Physics-to-Code Map



\### Ket

A ket is a state vector.



In code, this is usually a NumPy array like:

\- `|H⟩`

\- `|V⟩`

\- `|D⟩`

\- `|R⟩`



The functions in `states.py` return these vectors.



\### Projector

A projector is built from a ket as



`|ψ⟩⟨ψ|`



In code, this is a matrix built from the outer product of a state vector with its conjugate transpose.



Projectors are used as:

\- pure-state density matrices

\- measurement operators

\- target states for fidelity calculations



\### Density Matrix

A density matrix is the matrix representation of a quantum state.



For pure states, it is just the projector of the ket.

For mixed states, it encodes a statistical ensemble.



Tomography aims to reconstruct this object from measurement data.



\### Tensor Product

A tensor product combines subsystems into a joint state.



Examples:

\- one qubit: `|H⟩`

\- another qubit: `|V⟩`

\- joint state: `|HV⟩`



This is essential for two-photon and two-qubit analysis.



\### Bell State

A Bell state is an ideal maximally entangled two-qubit state.



Examples:

\- `Φ+`

\- `Φ-`

\- `Ψ+`

\- `Ψ-`



These are the main targets in the Bell-analysis part of the project.



\### Fidelity

Fidelity measures how close one quantum state is to another.



In this repo, fidelity is especially important for:

\- comparing reconstructed states to ideal Bell states

\- evaluating tomography results

\- judging state-preparation quality



\### Tomography

Tomography reconstructs a density matrix from measurement data.



In this project, tomography will eventually take:

\- measurement counts

\- basis labels

\- projectors / measurement settings



and output:

\- a reconstructed density matrix

\- physicality-cleaned state estimates

\- derived metrics such as purity and fidelity



\### CHSH / Bell Analysis

Bell analysis asks whether the reconstructed or measured state shows nonclassical correlations.



This part of the repo will eventually:

\- compute correlation functions

\- compute CHSH S

\- optimize analyzer settings

\- compare theoretical and measured violation



\## Current Status



Currently implemented:

\- `states.py`

\- `metrics.py`



Currently planned next:

\- `bell.py`

\- `chsh.py`

\- `tomography.py`

\- `mle.py`



The repo is still in an early build phase. Right now the focus is on extracting reusable logic from the original notebooks and reorganizing it into a clean package.



\## How To Use This Repository



The intended workflow is:



1\. keep reusable physics code in `src/pec`

2\. keep tests in `tests`

3\. keep demonstrations in `notebooks`

4\. use Git to save each clean step

5\. use GitHub as the public and portfolio-facing home of the project



\## Project Goal



The ultimate goal is not just to preserve coursework.



The goal is to turn notebook-based lab work into a package that demonstrates:



\- understanding of photonic QIP experiments

\- clean Python implementation

\- reproducible scientific workflows

\- meaningful state characterization

\- a bridge between theory, experiment, and software



That is what gives this repository value as both a technical tool and a portfolio project.



