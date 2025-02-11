import os
import time
import numpy as np
import pandas as pd
import shap
import joblib
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE, LogScore
from ngboost.learners import default_tree_learner
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Setup
areas = [
    "Nordic",
    "CE",
]
data_version = "2024-05-19"
targets = ["f_integral", "f_ext", "f_msd", "f_rocof"]
model_type = "_full"
model_name = "ngb"
scaler = "yeo_johnson"
scaled_str = "_scaled"

start_time = time.time()

for area in areas:
    print(f"---------------------------- {area} ------------------------------------")
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
        y_pred["daily_profile"] = [daily_profile[time] for time in X_test.index.time]

        # Mean predictor
        y_pred["mean_predictor"] = y_train.mean()
        y_pred["std_predictor"] = y_train.std()

        # Split training set into training and validation sets
        X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Parameters for hyper-parameter optimization
        param_grid = {
            "n_estimators": [500, 650, 800],
            "learning_rate": [0.001, 0.01, 0.05, 0.1],
            "Base": [default_tree_learner],
            "Dist": [Normal],
            "Score": [MLE, LogScore],
        }

        # Grid search for optimal hyper-parameters
        grid_search = GridSearchCV(
            NGBRegressor(natural_gradient=True, random_state=42),
            param_grid,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            verbose=1,
            n_jobs=25,
            cv=5,
        )

        grid_search.fit(
            X_train_train,
            y_train_train,
            X_val=X_train_val,
            Y_val=y_train_val,
            early_stopping_rounds=10,
        )

        # Save CV results
        pd.DataFrame(grid_search.cv_results_).to_csv(
            f"{res_folder}cv_results_ngb{model_type}.csv"
        )

        # Save best params
        best_params = grid_search.best_estimator_.get_params()
        best_params["n_estimators"] = grid_search.best_estimator_.n_estimators
        pd.DataFrame(best_params, index=[0]).to_csv(
            f"{res_folder}cv_best_params_ngb{model_type}.csv"
        )

        # Train best model on whole training set
        model = grid_search.best_estimator_
        model.fit(X_train, y_train)
        joblib.dump(model, f"{res_folder}best_ngb_model.pkl")

        # Calculate predictions and confidence intervals
        y_test_dist = model.pred_dist(X_test)
        y_pred["ngb_prediction"] = y_test_dist.loc
        y_pred["ngb_upper"] = y_test_dist.dist.interval(0.95)[1]
        y_pred["ngb_lower"] = y_test_dist.dist.interval(0.95)[0]

        print(
            f"{model_type[1:]} Best performance: {r2_score(y_test, y_pred['ngb_prediction'])}"
        )

        # Calculate SHAP values on test set
        if (area in ["CE", "Nordic"]) and (model_type == "_full"):
            # Mean SHAP values
            shap_vals_mean = shap.TreeExplainer(model, model_output=0).shap_values(X_test)
            np.save(f"{res_folder}shap_values_mean{model_type}.npy", shap_vals_mean)
            
            # Standard deviation SHAP values
            shap_vals_std = shap.TreeExplainer(model, model_output=1).shap_values(X_test)
            np.save(f"{res_folder}shap_values_std{model_type}.npy", shap_vals_std)


        # Save predictions
        y_pred.to_hdf(f"{res_folder}y_pred.h5", key="df")

print(f"Execution time [h]: {(time.time() - start_time) / 3600.0}")