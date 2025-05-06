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

The `scripts` contains 4 scripts for model training and 4 scripts for obtaining explanations:
The `model_fit_$.py` scripts contain code for fitting the model  and optimize hyper-parameters.
The `explain_$_partition.py` scripts contain code for obtaining explanations.


## Input data and results


