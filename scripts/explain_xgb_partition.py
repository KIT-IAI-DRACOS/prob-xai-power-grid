import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from typing import List, NamedTuple
import sys

sys.path.append("./")
sys.path.append("..")

import os
import numpy as np
import pandas as pd
import shap
import typer
import xgboost as xgb

import scipy.cluster.hierarchy as sch


np.random.seed(42)

scaler = "yeo_johnson"
data_version = "2024-05-19"
results_version = "2024-05-19"
model_type = "_full"
areas = ["CE", "Nordic"]
targets = ["f_rocof", "f_ext"]

for area in areas:
    print(f"---------------------------- {area} ------------------------------------")

    data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"

    for target in targets:
        print(f"-------- {target} --------")
        data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"
        fit_folder = f"../results/model_fit/{area}/version_{results_version}_xgb/target_{target}/"

        scaled_str = "_scaled"
        res_folder = f"../results/model_fit/{area}/version_{results_version}_xgb/target_{target}/explanations_partition/"

        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        X_train = pd.read_hdf(data_folder + f"X_train{model_type}{scaled_str}.h5")
        y_train = pd.read_hdf(data_folder + f"y_train{scaled_str}.h5").loc[:,target]
        X_test = pd.read_hdf(data_folder + f"X_test{model_type}{scaled_str}.h5")
        y_test = pd.read_hdf(data_folder + f"y_test{scaled_str}.h5").loc[:,target]
        y_pred = pd.read_hdf(data_folder + "y_pred.h5")  # contains only time index

        best_params = pd.read_csv(fit_folder + 'cv_best_params_gtb_full.csv',
                    usecols=['max_depth', 'learning_rate', 'subsample', 'min_child_weight', 'reg_lambda', 
                            'n_estimators', 'base_score', 'objective']
                )
        best_params = best_params.to_dict("records")[0]
        best_params["n_jobs"] = 25
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train.values)
        y_pred = model.predict(X_test)
        print(r2_score(y_test.values, y_pred))

        correlation_matrix = X_train.corr(method='pearson')
        # take absolute values
        correlation_matrix = np.abs(np.corrcoef(correlation_matrix))
        dist_matrix = 1 - correlation_matrix
        clustering = sch.linkage(dist_matrix, method="complete")
        masker = shap.maskers.Partition(X_train, clustering=clustering)


        mean_explainer = shap.PartitionExplainer(model.predict, masker)
        mean_shap_values = mean_explainer(X_test.values)
        
        np.save(
            res_folder + "shap_values_mean.npy",
            mean_shap_values.values,
        )
    
