# Probabilistic and Explainable Machine Learning for Tabular Power Grid Data

This repository contains the code for the paper "Probabilistic and Explainable Machine Learning for Tabular Power Grid Data".

## Installation

This project uses `uv` for dependency management and was written in `python3.11`.

### Prerequisites

- Python 3.9 or higher
- `uv` - ([Installation guide](https://github.com/astral-sh/uv))

### Setting up the environment

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Create and activate a virtual environment with `uv`
    ```bash
        uv venv
        source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. Install dependencies
    ```bash
        uv pip install -e . 
    ```


## Code structure

The `scripts` folder contains scripts to create the paper results and `notebooks` contains a notebook to reproduce the paper figures and evaluate the results. The `utils` folder comprises scripts for TabNetProba as well as plotting modules.

The `scripts` folder contain a pipeline of six different stages:

* `1_download_data.sh`: A bash script to download the external features from the ENTSO-E Transparency Platform. 
* `2_stability_indicator_prep.py`: Create HDF files from grid frequency CSV file and then extract frequency stability indicators.
* `3_entsoe_data_prep.py`: Collect and aggregate external features within each synchronous area.
* `4_external_feature_prep.py`: Add additional engineered features to the set of external features.
* `5_train_test_split.py`: Split data set into train and test set and save data in a version folder.
* `model_fit_*.py` - Scripts for fitting models and optimizing hyperparameters
* `explain_*_partition.py` - Scripts for obtaining explanations for each model.
 

Example usage:
```bash
uv run scripts/model_fit_model1.py  # Train model 1
uv run scripts/explain_model1_partition.py  # Generate explanations for model 1
```

The `notebooks` directory contains:
    * `clustering.ipynb` - Clustering analysis methods and results
    * `data_overview.ipynb` - Overview and exploration of the dataset
    * `eval_probabilistic.ipynb` - Probabilistic evaluation methodology
    * `model_performance.ipynb` - Performance analysis of trained models
    * `model_runtimes.ipynb` - Runtime benchmarks for different models
    * `plots_for_paper.ipynb` - Visualization code for publication figures
    * `standardize_data.ipynb` - Data standardization procedures


## Input data and results

* **External features, frequency stability indicators and results of hyper-parameter optimization and model interpretation**: The output of scripts 2 to 5 and the model fit and explanation scripts is assumed to reside in the repository directory within `./data/` and the results should reside in `./results/`. In particular, the data of external features and stability indicators can be used to re-run the model fit. 
* **Raw grid frequency data**: We have used pre-processed [grid frequency data](https://zenodo.com) as an input to 2_stability_indicator_prep.py. The CSV files from the repository are assumed to reside in `../Frequency_data_base/` relative to this code repository. The frequency data is originally based on publicly available measurements from the [German TSOs](https://www.netztransparenz.de/de-de/Regelenergie/Daten-Regelreserve/Sek%C3%BCndliche-Daten).
* **Raw ENTSO-E data**: The output of `1_download_data.sh` can be downloaded from the [ENTSO-E Transparency Platform](transparency.entsoe.eu/) via the bash script. The ENTSO-E data is assumed to reside in `../../External_data/ENTSO-E` relative to this code repository.


