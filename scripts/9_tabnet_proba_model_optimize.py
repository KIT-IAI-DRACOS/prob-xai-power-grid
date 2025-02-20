import os
import typer

import pandas as pd
import numpy as np
import torch
import sys

sys.path.append("./")
sys.path.append("..")

from sklearn.model_selection import train_test_split
from models.tabnet_proba import TabNetRegressorProba
from random import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

np.random.seed(42)

data_version = "2024-05-19"
results_version = "2024-08-13"
model_type = "_full"
# targets = ["f_rocof", "f_integral", "f_ext", "f_msd"]


def run_model(area: str, target: str, scaler: str):
    # Retrain the model with the best hyperparameters on the full training data
    data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"
    res_folder = f"../results/model_fit/{area}/version_{results_version}_tabnet_proba_opt/target_{target}/"

    if scaler is None:
        scaled_str = ""
        res_folder = f"../results/model_fit/{area}/version_{results_version}_tabnet_proba_opt/target_{target}/"
    else:
        # Result folder where prediction, SHAP values and CV results are saved
        scaled_str = "_scaled"
        res_folder = f"../results/model_fit/{area}/version_{results_version}_tabnet_proba_opt/{scaler}/target_{target}/"

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    y_train = pd.read_hdf(data_folder + f"y_train{scaled_str}.h5").loc[:, target]
    y_test = pd.read_hdf(data_folder + f"y_test{scaled_str}.h5").loc[:, target]
    y_pred = pd.read_hdf(data_folder + "y_pred.h5")  # contains only time index

    # Load feature data
    X_train = pd.read_hdf(data_folder + f"X_train{model_type}{scaled_str}.h5")
    X_test = pd.read_hdf(data_folder + f"X_test{model_type}{scaled_str}.h5")

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Daily profile prediction
    daily_profile = y_train.groupby(X_train.index.time).mean()
    daily_profile_std = y_train.groupby(X_train.index.time).std()
    y_pred["daily_profile"] = [daily_profile[time] for time in X_test.index.time]
    y_pred["daily_profile_std"] = [
        daily_profile_std[time] for time in X_test.index.time
    ]

    y_pred["mean_predictor"] = y_train.mean()
    y_pred["std_predictor"] = y_train.std()
    y_pred[f"{target}"] = y_test

    search_space = {
        "n_d": lambda: randint(8, 64),
        "n_a": lambda: randint(8, 64),
        "n_steps": lambda: randint(3, 10),
        "n_independent": lambda: randint(3, 10),
        "gamma": lambda: uniform(3, 10),
        "n_shared": lambda: randint(3, 10),
    }

    # Number of random search iterations
    n_iterations = 100
    results = []

    # First, train model with default parameters
    model = TabNetRegressorProba(cat_dims=[], cat_emb_dim=[], cat_idxs=[])
    model.fit(
        X_train=X_train.values,
        y_train=y_train.values.reshape(-1, 1),
        max_epochs=100,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )

    preds = model.predict(X_train_val.values)
    loss = float(
        model.compute_loss(
            torch.tensor(preds), torch.tensor(y_train_val.values.reshape(-1, 1))
        )
    )
    filtered_params = {k: v for k, v in model.get_params().items() if k in search_space}

    results.append({"params": filtered_params, "loss": loss})

    # Random search loop
    for i in range(n_iterations):
        # Sample hyperparameters
        params = {key: fn() for key, fn in search_space.items()}
        print(f"Iteration {i}: Sampled Params - {params}")

        # Instantiate the model with sampled hyperparameters
        model = TabNetRegressorProba(
            n_d=params["n_d"],
            n_a=params["n_a"],
            n_steps=params["n_steps"],
            n_independent=params["n_independent"],
            gamma=params["gamma"],
            n_shared=params["n_shared"],
        )

        model.fit(
            X_train_train.values,
            y_train_train.values.reshape(-1, 1),
            max_epochs=100,
            patience=50,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=10,
            drop_last=False,
        )

        preds = model.predict(X_train_val.values)
        loss = float(
            model.compute_loss(
                torch.tensor(preds), torch.tensor(y_train_val.values.reshape(-1, 1))
            )
        )

        results.append({"params": params, "loss": loss})

    # Find the best parameters
    best_result = min(results, key=lambda x: x["loss"])
    print("Best hyperparameters found: ", best_result["params"])
    print("Corresponding loss: ", best_result["loss"])

    best_params = best_result["params"]

    # Instantiate the model with the best hyperparameters
    best_model = TabNetRegressorProba(
        n_d=best_params["n_d"],
        n_a=best_params["n_a"],
        n_steps=best_params["n_steps"],
        n_independent=best_params["n_independent"],
        gamma=best_params["gamma"],
        n_shared=best_params["n_shared"],
        # scheduler_fn=torch.optim.lr_scheduler.StepLR,
    )

    # Train the model on the full dataset
    best_model.fit(
        X_train.values,
        y_train.values.reshape(-1, 1),
        max_epochs=100,
        patience=50,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=10,
        drop_last=False,
    )

    res_folder_model = res_folder + "best_model_params"

    if not os.path.exists(res_folder_model):
        os.makedirs(res_folder_model)
    best_model.save_model(res_folder_model)
    preds = best_model.predict(X_test.values)
    y_pred[f"{target}_prediction"] = preds[:, 0]
    y_pred[f"{target}_cov_pred"] = preds[:, 1]
    y_pred.to_hdf(res_folder + "y_pred.h5", key="df")


if __name__ == "__main__":
    typer.run(run_model)
