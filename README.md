# git remote add origin git@github.com:KIT-IAI-DRACOS/prob-xai-power-grid.git

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

The `scripts` directory contains:
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


