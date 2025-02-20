import os
import time
import json
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from random import randint, uniform, choice

sys.path.append("./")
sys.path.append("..")
from utils.tabnet_proba import TabNetRegressorProba

np.random.seed(42)

areas = [
    "Nordic",
    #"CE",
]
data_version = "2024-05-19"
targets = ["f_integral", 
           #"f_ext", "f_msd", "f_rocof"
           ]
model_type = "_full"
model_name = "tabnet_proba"
scaler = "yeo_johnson"
scaled_str = "_scaled"

for area in areas:
    print(f"---------------------------- {area} ------------------------------------")

    data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"

    for target in targets:
        print(f"-------- {target} --------")

        # Result folder for predictions and optimization results
        res_folder = f"../results/model_fit/{area}/version_{data_version}_{model_name}/{scaler}/target_{target}/"
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        # Load training and test data
        y_train = pd.read_hdf(data_folder + f"y_train{scaled_str}.h5").loc[:, target]
        y_test = pd.read_hdf(data_folder + f"y_test{scaled_str}.h5").loc[:, target]
        y_pred = pd.read_hdf(data_folder + "y_pred.h5")  # contains only time index
        X_train = pd.read_hdf(data_folder + f"X_train{model_type}{scaled_str}.h5")
        X_test = pd.read_hdf(data_folder + f"X_test{model_type}{scaled_str}.h5")

        # Split training data for validation
        X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Compute daily profile predictions
        daily_profile = y_train.groupby(X_train.index.time).mean()
        daily_profile_std = y_train.groupby(X_train.index.time).std()
        y_pred["daily_profile"] = [daily_profile[time] for time in X_test.index.time]
        y_pred["daily_profile_std"] = [daily_profile_std[time] for time in X_test.index.time]
        y_pred["mean_predictor"] = y_train.mean()
        y_pred["std_predictor"] = y_train.std()
        y_pred[f"{target}"] = y_test

        search_space = {
            "n_d": lambda: randint(8, 64),
            "n_a": lambda: randint(8, 64),
            "n_steps": lambda: randint(3, 10),
            "n_independent": lambda: randint(1, 5),
            "gamma": lambda: uniform(1, 2),
            "momentum": lambda: 10 ** uniform(-2, -0.4),
            "lambda_sparse": lambda: 10 ** uniform(-5, -3),
            "mask_type": lambda: choice(["sparsemax", "entmax"]),
            "learning_rate": lambda: 10 ** uniform(-4, -2),
            "epochs": lambda: choice([100, 150, 200, 250, 300, 350, 400, 450, 500]),
        }

        n_iterations = 1
        results = []

        # Initial model training with default parameters
        std_model = TabNetRegressorProba(cat_dims=[], cat_emb_dim=[], cat_idxs=[])
        std_model.fit(
            X_train=X_train_train.values,
            y_train=y_train_train.values.reshape(-1, 1),
            max_epochs=100,
            patience=30,
            batch_size=1024,
            virtual_batch_size=128,
            #num_workers=1,
            drop_last=False,
        )
        
        preds = std_model.predict(X_train_val.values)
        loss = float(
            std_model.compute_loss(
                torch.tensor(preds), torch.tensor(y_train_val.values.reshape(-1, 1))
            )
        )
        filtered_params = {k: v for k, v in std_model.get_params().items() if k in search_space}

        results.append({"params": filtered_params, "loss": loss})

        # Hyperparameter tuning with random search
        for i in range(n_iterations):
            params = {key: fn() for key, fn in search_space.items()}
            print(f"Iteration {i}: Sampled Params - {params}")

            test_model = TabNetRegressorProba(
                cat_dims=[],
                cat_emb_dim=[],
                cat_idxs=[],
                n_d=params["n_d"],
                n_a=params["n_d"],
                n_independent=params["n_independent"],
                gamma=params["gamma"],
                momentum=params["momentum"],
                lambda_sparse=params["lambda_sparse"],
                mask_type=params["mask_type"],
                optimizer_params=dict(lr=params["learning_rate"]),
                seed=42,
            )

            test_model.fit(
                X_train_train.values,
                y_train_train.values.reshape(-1, 1),
                max_epochs=params["epochs"],
                patience=30,
                batch_size=1024,
                virtual_batch_size=128,
                #num_workers=1,
                drop_last=False,
            )

            preds = test_model.predict(X_train_val.values)
            loss = float(
                test_model.compute_loss(
                    torch.tensor(preds), torch.tensor(y_train_val.values.reshape(-1, 1))
                )
            )

            results.append({"params": params, "loss": loss})

        # Select the best parameters based on loss
        best_result = min(results, key=lambda x: x["loss"])
        print("Best hyperparameters found: ", best_result["params"])
        print("Corresponding loss: ", best_result["loss"])

        # Save results to JSON
        with open(res_folder + f"{randint(0, 9999999999)}.json", "w") as f:
            json.dump(results, f, indent=2)

        # Train the best model on the full dataset using best hyperparameters
        best_params = best_result["params"]
        best_model = TabNetRegressorProba(
            n_d=best_params["n_d"],
            n_a=best_params["n_a"],
            n_independent=best_params["n_independent"],
            gamma=best_params["gamma"],
            momentum=best_params["momentum"],
            lambda_sparse=best_params["lambda_sparse"],
            mask_type=best_params["mask_type"],
            optimizer_params=dict(lr=best_params["learning_rate"]),
            seed=42,
        )

        best_model.fit(
            X_train.values,
            y_train.values.reshape(-1, 1),
            max_epochs=best_params["epochs"],
            patience=50,
            batch_size=1024,
            virtual_batch_size=128,
            #num_workers=1,
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

