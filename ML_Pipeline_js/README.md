# QSO-LAE Lens Search – Machine Learning Pipeline

This directory contains a machine learning pipeline developed by Jonas Spiller (ETH Zurich).

The pipeline is designed to identify quasars (QSOs) acting as strong gravitational lenses for background Lyman-Alpha Emitters (LAEs) in the DESI DR1 dataset. It builds upon and extends the pipeline developed by McArthur et al. (2025) by adapting it to higher redshifts and using simulated LAE emission profiles (instead of ELG spectra).


---

## 📁 Structure

```bash
.
├── LensCreation_LAE.ipynb  # Creation of QSO-LAE mock lenses. **Note**: This repo **does not include** the LAE augmentation pipeline, generating the LAE spectra used to create mock lenses. For that, see [augmented_LAE_4QSOlensing](https://github.com/JonasSpiller/augmented_LAE_4QSOlensing) (private)
├── LensCreation_LAE.py     # Creation of QSO-LAE mock lenses as .py file.
├── main.ipynb              # Entry point: Data loading, model selection and starting pipeline
├── main.py                 # Main skript as .py file.
├── utility_functions.py    # Organization of hyperparameter optimization and training.
├── MLmodels.py             # CNN model architectures
├── HPtuning_GA.py          # Genetic algorithm for hyperparameter optimization
├── results.ipynb           # Results and model evaluation 
