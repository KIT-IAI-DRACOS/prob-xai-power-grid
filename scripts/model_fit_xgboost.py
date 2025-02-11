import os
import time

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

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
model_name = "xgb"
scaler = "yeo_johnson"
scaled_str = "_scaled"

start_time = time.time()

for area in areas:

    print(
        f"---------------------------- {area} ------------------------------------"
    )

    data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"

    for target in targets:

        print(f"-------- {target} --------")

        # Result folder where prediction, SHAP values and CV results are saved
        res_folder = f"../results/model_fit/{area}/version_{data_version}_{model_name}/{scaler}/target_{target}/"

        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        # Load target data
        y_train = pd.read_hdf(data_folder + f"y_train{scaled_str}.h5").loc[:, target]
        y_test = pd.read_hdf(data_folder + f"y_test{scaled_str}.h5").loc[:, target]
        y_pred = pd.read_hdf(data_folder + "y_pred.h5")  # contains only time index

        # Load feature data
        X_train = pd.read_hdf(data_folder + f"X_train{model_type}{scaled_str}.h5")
        X_test = pd.read_hdf(data_folder + f"X_test{model_type}{scaled_str}.h5")

        # Daily profile prediction
        daily_profile = y_train.groupby(X_train.index.time).mean()
        y_pred["daily_profile"] = [
            daily_profile[time] for time in X_test.index.time
        ]

        # Mean predictor
        y_pred["mean_predictor"] = y_train.mean()

        #### Gradient boosting Regressor CV hyperparameter optimization ###

        # Split training set into (smaller) training set and validation set
        X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
            X_train, y_train, test_size=0.2
        )
        # Parameters for hyper-parameter optimization
        params_grid = {
            "max_depth": [3, 5, 7, 9, 11],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [1, 0.7, 0.4],
            "min_child_weight": [1, 5, 10, 30],
            "reg_lambda": [0.1, 1, 10],
        }
        
        fit_params = {
            "eval_set": [
                (X_train_train, y_train_train),
                (X_train_val, y_train_val),
            ],
            "verbose": 0,
        }

        # Grid search for optimal hyper-parameters
        grid_search = GridSearchCV(
            xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1000,
                verbosity=0,
                n_jobs=1,
                base_score=y_train.mean(),
                early_stopping_rounds=20,
            ),
            params_grid,
            verbose=1,
            n_jobs=25,
            cv=5,
        )

        grid_search.fit(X_train_train, y_train_train, **fit_params)

        # Save CV results
        pd.DataFrame(grid_search.cv_results_).to_csv(
            f"{res_folder}cv_results_gtb{model_type}.csv"
        )

        # Save best params (including n_estimators from early stopping on validation set)
        best_params = grid_search.best_estimator_.get_params()

        # best_params["n_estimators"] = grid_search.best_estimator_.best_ntree_limit
        best_params["n_estimators"] = grid_search.best_estimator_.n_estimators
        pd.DataFrame(best_params, index=[0]).to_csv(
            f"{res_folder}cv_best_params_gtb{model_type}.csv"
        )

        # Gradient boosting regression best model evaluation on test set
        best_params = pd.read_csv(
            f"{res_folder}cv_best_params_gtb{model_type}.csv",
            usecols=list(params_grid.keys())
            + ["n_estimators", "base_score", "objective"],
        )
        best_params = best_params.to_dict("records")[0]
        best_params["n_jobs"] = 25
        print(f"Number of opt. boosting rounds: {best_params['n_estimators']}")

        # Train on whole training set (including validation set)
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        # Calculate SHAP values on test set
        if (area in [
            "CE",
            "Nordic",
            # "GB"
        ]) and (model_type == "_full"):
            shap_vals = shap.TreeExplainer(model).shap_values(X_test)
            np.save(
                f"{res_folder}shap_values_gtb{model_type}.npy",
                shap_vals,
            )

        # Prediction on test set
        y_pred[f"gtb{model_type}"] = model.predict(X_test)
        print(
            f"{model_type[1:]} Best performance: {r2_score(y_test, y_pred[f'gtb{model_type}'])}"
        )

    # Save prediction
    y_pred.to_hdf(f"{res_folder}y_pred.h5", key="df")


print(f"Execution time [h]: {(time.time() - start_time) / 3600.0}")