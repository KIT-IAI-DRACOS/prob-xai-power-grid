import os

import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import PowerTransformer

# Choose subset of targets if required
targets = ["f_integral", "f_rocof", "f_ext", "f_msd"]

# Areas inlcuding "country" areas
areas = [
    "CE",
    # "GB",
    "Nordic",
    "SE",
    "CH",
    "DE",
]


for area in areas:

    print("Processing external features from", area)

    # Setup folder for this specific version of train-test data
    folder = "../data/2020-2024/{}/".format(area)
    norm_data_folder = folder + "yeo_johnson" + "/"
    version_folder = (
        folder + "version_" + "2024-05-19/"
    )  # pd.Timestamp("today").strftime("%Y-%m-%d") + '/'
    if not os.path.exists(version_folder):
        os.makedirs(version_folder)

    # Load actual and forecast (day-ahead available) data
    X_actual = pd.read_hdf(folder + "input_actual.h5")
    X_forecast = pd.read_hdf(folder + "input_forecast.h5")
    y = pd.read_hdf(folder + "outputs.h5").loc[:, targets]

    # Drop nan values
    valid_ind = ~pd.concat([X_forecast, X_actual, y], axis=1).isnull().any(axis=1)
    X_forecast, X_actual, y = X_forecast[valid_ind], X_actual[valid_ind], y[valid_ind]

    # Join features for full model
    X_full = X_actual.join(X_forecast)

    # Train-test split
    X_train_full, X_test_full, y_train, y_test = model_selection.train_test_split(
        X_full, y, test_size=0.2, random_state=42
    )
    X_train_day_ahead = X_forecast.loc[X_train_full.index]
    X_test_day_ahead = X_forecast.loc[X_test_full.index]
    y_pred = pd.DataFrame(index=y_test.index)

    # Save data for full model and restricted (day-ahead) model
    X_train_full.to_hdf(version_folder + "X_train_full.h5", key="df")
    X_train_day_ahead.to_hdf(version_folder + "X_train_day_ahead.h5", key="df")
    y_train.to_hdf(version_folder + "y_train.h5", key="df")
    y_test.to_hdf(version_folder + "y_test.h5", key="df")
    y_pred.to_hdf(version_folder + "y_pred.h5", key="df")
    X_test_full.to_hdf(version_folder + "X_test_full.h5", key="df")
    X_test_day_ahead.to_hdf(version_folder + "X_test_day_ahead.h5", key="df")

    # Create data scalers
    scaler_X = PowerTransformer(method="yeo-johnson")
    scaler_y = PowerTransformer(method="yeo-johnson")
    
    # Scale the full feature set
    X_train_scaled = scaler_X.fit_transform(X_train_full)
    X_test_scaled = scaler_X.transform(X_test_full)
    
    # Scale the targets
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert back to DataFrames
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=X_train_full.columns, index=X_train_full.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=X_test_full.columns, index=X_test_full.index
    )
    y_train_scaled_df = pd.DataFrame(
        y_train_scaled, columns=y_train.columns, index=y_train.index
    )
    y_test_scaled_df = pd.DataFrame(
        y_test_scaled, columns=y_test.columns, index=y_test.index
    )
    
    # Save the scaled data
    X_train_scaled_df.to_hdf(
        norm_data_folder + "X_train_full_scaled.h5", key="X_train_scaled", mode="w"
    )
    X_test_scaled_df.to_hdf(
        norm_data_folder + "X_test_full_scaled.h5", key="X_test_scaled", mode="w"
    )
    y_train_scaled_df.to_hdf(
        norm_data_folder + "y_train_scaled.h5", key="y_train_scaled", mode="w"
    )
    y_test_scaled_df.to_hdf(
        norm_data_folder + "y_test_scaled.h5", key="y_test_scaled", mode="w"
    )
    y_pred.to_hdf(norm_data_folder + "y_pred.h5", key="y_pred", mode="w")
    
    print(f"Successfully processed and scaled data for {area}")
