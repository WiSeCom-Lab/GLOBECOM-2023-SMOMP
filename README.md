# Paper code
Code to run the simulations and create the figures for the ICASSP paper "Separable multidimensional orthogonal matching pursuit and its application to joint localization and communication at mmWave"

# Simulation
The ray tracing scenario has already been run and the simulated results are saves in the `data/office-walls` folder.
The channel decomposition for these simulations can be run from the script `src/decompose_HH.py`, which will save the channel decomposition results in `data/office-walls-HH/paths`.
For convenience, these results are already stored in this folder.

# Results
Before proceeding, you will probably have to create the folder `data/office-walls-HH/figures` to avoid any folder permission problems with the next scripts.

### Localization
The localization algorithm can be run with `src/localize_single_multiplePt-budget.py` and will generate and save the corresponding figure.

### Spectral efficiency
The spectral efficiency can be computed with `src/spectral_multiplePt-budget.py` and will generate and save the corresponding figure.