import os
import time

import numpy as np
import pandas as pd
import optuna
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold

# Setup
areas = [
    # "DE",
    # "GB",
    "Nordic",
    "CE",
    # "SE",
    # "CH",
]
data_version = "2024-05-19"
targets = ["f_integral", "f_ext", "f_msd", "f_rocof"]
model_type = "_full"
model_name = "tabnet"
scaler = "yeo_johnson"
scaled_str = "_scaled"

start_time = time.time()

def tabnet_objective(trial, X_train, y_train):
    mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_da = trial.suggest_int("n_da", 8, 64, step=4)
    gamma = trial.suggest_float("gamma", 1., 2., step=0.1)
    momentum = trial.suggest_float("momentum", 1e-6, 1e-3, log=True)
    lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    patienceScheduler = trial.suggest_int("patienceScheduler", low=3, high=10)
    patience = trial.suggest_int("patience", low=15, high=30)
    epochs = trial.suggest_int('epochs', 100, 500, step=50)

    tabnet_params = dict(
        n_d=n_da, n_a=n_da,  gamma=gamma, momentum=momentum,
        lambda_sparse=lambda_sparse, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        mask_type=mask_type, 
        scheduler_params=dict(mode="min", patience=patienceScheduler, min_lr=1e-5, factor=0.5,),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        verbose=0,
    )

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    CV_score_array = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
        model = TabNetRegressor(**tabnet_params)
        model.fit(
            X_train=X_train_fold, y_train=y_train_fold, eval_set=[(X_val_fold, y_val_fold)],
            patience=patience, max_epochs=epochs, eval_metric=['mse']
        )
        CV_score_array.append(model.best_cost)

    return np.mean(CV_score_array)

for area in areas:
    print(f"---------------------------- {area} ------------------------------------")

    data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"

    for target in targets:
        print(f"-------- {target} --------")

        # Result folder for predictions and optimization results
        res_folder = f"../results/model_fit/{area}/version_{data_version}_{model_name}/{scaler}/target_{target}/"
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        # Load target data
        y_train = pd.read_hdf(data_folder + f"y_train{scaled_str}.h5").loc[:, target].values
        y_test = pd.read_hdf(data_folder + f"y_test{scaled_str}.h5").loc[:, target].values
        y_pred = pd.read_hdf(data_folder + "y_pred.h5")  # contains only time index

        # Load feature data
        X_train = pd.read_hdf(data_folder + f"X_train{model_type}{scaled_str}.h5").values
        X_test = pd.read_hdf(data_folder + f"X_test{model_type}{scaled_str}.h5").values

        # Optimize hyperparameters using Optuna
        study = optuna.create_study(direction="minimize", study_name=f'TabNet optimization {area} {target}')
        study.optimize(lambda trial: tabnet_objective(trial, X_train, y_train.reshape(-1,1)), timeout=6*60)  # 5 hours
        #study.optimize(lambda trial: tabnet_objective(trial, X_train, y_train.reshape(-1,1)), n_trials=1) 
        
        # Get best parameters
        best_params = study.best_params
        final_params = dict(
            n_d=best_params['n_da'], n_a=best_params['n_da'], gamma=best_params['gamma'],
            lambda_sparse=best_params['lambda_sparse'], optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type=best_params['mask_type'], 
            scheduler_params=dict(mode="min", patience=best_params['patienceScheduler'], min_lr=1e-5, factor=0.5,),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0,
        )
        epochs = best_params['epochs']

        # Train final model on full training set
        model = TabNetRegressor(**final_params)
        model.fit(X_train=X_train, y_train=y_train.reshape(-1,1), patience=best_params['patience'], max_epochs=epochs, eval_metric=['mse'])
        res_folder_model = res_folder + "best_model_params"
    
        if not os.path.exists(res_folder_model):
            os.makedirs(res_folder_model)
        model.save_model(res_folder_model)
        # Save best parameter results
        pd.DataFrame([best_params]).to_csv(f"{res_folder}optuna_best_params_tabnet{model_type}.csv", index=False)

        # Predict on test set
        y_pred[f"{model_type}"] = model.predict(X_test)

        # Evaluate and print R^2 score
        print(f"{model_type[1:]} Best performance: {r2_score(y_test, y_pred[f'tabnet{model_type}'])}")

    # Save predictions
    y_pred.to_hdf(f"{res_folder}y_pred.h5", key="df")

print(f"Execution time [h]: {(time.time() - start_time) / 3600.0}")