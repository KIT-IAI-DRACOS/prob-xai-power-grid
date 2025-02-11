import os
import typer

import pandas as pd
import numpy as np
import shap
import joblib

from scipy.stats import uniform, randint
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE, LogScore
from ngboost.learners import default_tree_learner

# from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
np.random.seed(42)

data_version = "2024-05-19"
results_version = "2024-06-24"
model_type = "_full"
# targets = ["f_rocof", "f_integral", "f_ext", "f_msd"]


def run_model(area: str, target: str, scaler: str):
    data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"
    print("-------- ", target, " --------")

    if scaler is None:
        scaled_str = ""
        res_folder = f"../results/model_fit/{area}/version_{results_version}_ngb/target_{target}/"
    else:
        # Result folder where prediction, SHAP values and CV results are saved
        scaled_str = "_scaled"
        res_folder = f"../results/model_fit/{area}/version_{results_version}_ngb/{scaler}/target_{target}/"

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    y_train = pd.read_hdf(data_folder + f"y_train{scaled_str}.h5").loc[:, target]
    y_test = pd.read_hdf(data_folder + f"y_test{scaled_str}.h5").loc[:, target]
    y_pred = pd.read_hdf(data_folder + "y_pred.h5")  # contains only time index

    # Load feature data
    X_train = pd.read_hdf(data_folder + f"X_train{model_type}{scaled_str}.h5")
    X_test = pd.read_hdf(data_folder + f"X_test{model_type}{scaled_str}.h5")

    # Daily profile prediction
    daily_profile = y_train.groupby(X_train.index.time).mean()
    y_pred["daily_profile"] = [daily_profile[time] for time in X_test.index.time]

    # Mean and std predictor + ground truth
    y_pred["mean_predictor"] = y_train.mean()
    y_pred["std_predictor"] = y_train.std()
    y_pred[f"{target}"] = y_test

    X_train_train, X_train_val, y_train_train, y_train_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [500, 650, 800],
        "learning_rate": [0.001, 0.01, 0.05, 0.1],
        "Base": [default_tree_learner],
        "Dist": [Normal],
        "Score": [MLE, LogScore],
    }

    model_ngb = NGBRegressor(natural_gradient=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model_ngb,
        param_grid=param_grid,
        scoring=make_scorer(mean_squared_error, greater_is_better=False),
        cv=3,
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(X_train_train, y_train_train)

    # Save CV results
    pd.DataFrame(grid_search.cv_results_).to_csv(
        res_folder + "cv_results_ngb{}.csv".format(model_type)
    )

    best_model = grid_search.best_estimator_
    best_params = best_model.get_params()
    best_params["n_estimators"] = best_model.n_estimators

    pd.DataFrame(best_params, index=[0]).to_csv(
        res_folder + "cv_best_params_ngb{}.csv".format(model_type)
    )

    best_model.fit(
        X_train_train,
        y_train_train,
        X_val=X_train_val,
        Y_val=y_train_val,
        early_stopping_rounds=10,
    )

    joblib.dump(best_model, res_folder + "best_ngb_model.pkl")

    y_test_ngb = best_model.pred_dist(X_test)
    y_pred["predictions"] = y_test_ngb.loc

    print(
        model_type[1:],
        "Best performance: {}".format(r2_score(y_test, y_pred["predictions"])),
    )

    y_pred["predictions_upper"] = y_test_ngb.dist.interval(0.95)[1]
    y_pred["predictions_lower"] = y_test_ngb.dist.interval(0.95)[0]
    # y_pred[f"{target}"] = y_test
    y_pred.to_hdf(res_folder + "y_pred.h5", key="df")

    # Calculate SHAP values on test set
    if area in [
        "CE",
        "Nordic",
    ]:
        if model_type == "_full":
            shap_vals_mean = shap.TreeExplainer(best_model, model_output=0).shap_values(
                X_test
            )
            np.save(
                f"{res_folder}shap_values_mean{model_type}.npy",
                shap_vals_mean,
            )

            shap_vals_std = shap.TreeExplainer(best_model, model_output=1).shap_values(
                X_test
            )
            np.save(
                f"{res_folder}shap_values_std{model_type}.npy",
                shap_vals_std,
            )
            shap_interact_vals_mean = shap.TreeExplainer(
                best_model, model_output=0
            ).shap_interaction_values(X_test)
            np.save(
                f"{res_folder}shap_interaction_values_mean{model_type}.npy",
                shap_interact_vals_mean,
            )
            shap_interact_vals_std = shap.TreeExplainer(
                best_model, model_output=1
            ).shap_interaction_values(X_test)
            np.save(
                f"{res_folder}shap_interaction_values_std{model_type}.npy",
                shap_interact_vals_std,
            )


if __name__ == "__main__":
    typer.run(run_model)
