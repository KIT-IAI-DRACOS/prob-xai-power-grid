import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from typing import List, NamedTuple
import sys

sys.path.append("./")
sys.path.append("..")
from pytorch_tabnet.tab_model import TabNetRegressor

import os
import numpy as np
import pandas as pd
import shap
import typer
import scipy.cluster.hierarchy as sch


np.random.seed(42)

data_version = "2024-05-19"
results_version = "2024-09-20"
model_type = "_full"
scaler = "yeo_johnson"
scaled_str = "_scaled"
areas = ["CE", "Nordic"]
targets = ["f_rocof", "f_ext"]

for area in areas:
    print(f"---------------------------- {area} ------------------------------------")

    data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"

    for target in targets:
        print(f"-------- {target} --------")
        data_folder = f"../data/2020-2024/{area}/version_{data_version}/{scaler}/"
        fit_folder = f"../results/model_fit/{area}/version_{results_version}_tabnet_det_rs/{scaler}/target_{target}/"

        scaled_str = "_scaled"
        res_folder = f"../results/model_fit/{area}/version_{results_version}_tabnet_det_rs/{scaler}/target_{target}/explanations_partition/"

        if not os.path.exists(res_folder):
            os.makedirs(res_folder)

        X_train = pd.read_hdf(data_folder + f"X_train{model_type}{scaled_str}.h5")
        X_test = pd.read_hdf(data_folder + f"X_test{model_type}{scaled_str}.h5")


        model = TabNetRegressor()
        model.load_model(filepath=fit_folder + "best_model_params.zip")

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