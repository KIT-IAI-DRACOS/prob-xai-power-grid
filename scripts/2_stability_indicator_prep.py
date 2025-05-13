import os
import sys

import numpy as np
import pandas as pd

sys.path.append("./")
sys.path.append("..")

from utils.stability_indicators import calc_rocof, make_frequency_data_hdf, fill_missing_seconds

# Time zones of frequency recordings
tzs = {"CE": "CET", "Nordic": "Europe/Helsinki", "GB": "Europe/London"}
# Datetime parameters for output data generation
start = pd.Timestamp("2020-01-01 00:00:00")
#end = pd.Timestamp("2020-12-30 23:59:59")  # for CE
end = pd.Timestamp("2023-12-31 23:59:59")
time_resol = pd.Timedelta("1h")

# Pre-processed frequency csv files
frequency_csv_folder = "../Frequency_data_base/"
tso_names = {"GB": "Nationalgrid", "CE": "TransnetBW", "Nordic": "Fingrid"}

# HDF frequency files (for faster access than csv files)
frequency_hdf_folder = {
    "GB": "../Frequency_data_preparation/Nationalgrid/",
    "CE": "../Frequency_data_preparation/TransnetBW/",
    "Nordic": "../Frequency_data_preparation/Fingrid/",
}

# Nan treatment
skip_hour_with_nan = True

# Parameters for rocof estimation
smooth_windows = {"CE": 60, "GB": 60, "Nordic": 30}
lookup_windows = {"CE": 60, "GB": 60, "Nordic": 30}

# GB To Do
for area in ["CE", "Nordic"]:  

    start = pd.Timestamp("2020-01-01 00:00:00")
    if area == "CE":
        end = pd.Timestamp("2023-12-30 23:59:59")
    else:
        end = pd.Timestamp("2023-12-31 23:59:59")

    print("\n######", area, "######")
    # If not existent, create HDF file from csv files
    # (for faster access when trying out things)
    hdf_file = make_frequency_data_hdf(
        frequency_csv_folder,
        tso_names[area],
        frequency_hdf_folder[area],
        start,
        end,
        tzs[area],
    )

    # Output data folder
    folder = "../data/2020-2024/{}/".format(area)

    if not os.path.exists(folder):
        os.makedirs(folder)

    # New: convert start and end
    start = pd.Timestamp("2020-01-01 00:00:00")
    # end = pd.Timestamp("2023-12-31 23:59:59")
    #end = pd.Timestamp("2023-12-30 23:59:59")
    if area == "CE":
        end = pd.Timestamp("2023-12-30 23:59:59")
    else:
        end = pd.Timestamp("2023-12-31 23:59:59")
    start = start.tz_localize(tzs[area])
    end = end.tz_localize(tzs[area])
    print(end)

    # Load frequency data
    freq = pd.read_hdf(hdf_file).loc[start:end]
    freq = freq - 50

    # fill missing seconds in the frequency data with nan
    freq = fill_missing_seconds(freq)

    # Setup datetime index for output data
    # index = pd.date_range(start, end, freq=time_resol, tz="UTC")
    index = pd.date_range(start, end, freq=time_resol, tz=tzs[area])
    index = index.tz_convert(tzs[area])
    if os.path.exists(folder + "outputs.h5"):
        outputs = pd.read_hdf(folder + "outputs.h5")
    else:
        outputs = pd.DataFrame(index=index)

    # Extract stability indicators
    print("Extracting stability indicators ...")
    outputs["f_integral"] = freq.groupby(pd.Grouper(freq="1h")).sum()
    outputs["f_ext"] = freq.groupby(pd.Grouper(freq="1h")).apply(
        lambda x: x[x.abs().idxmax()] if x.notnull().any() else np.nan
    )
    outputs["f_rocof"] = calc_rocof(freq, smooth_windows[area], lookup_windows[area])
    outputs["f_msd"] = (freq**2).groupby(pd.Grouper(freq="1h")).mean()

    # Set hour to NaN if frequency contains at least one NaN in that hour
    if skip_hour_with_nan == True:
        hours_with_nans = freq.groupby(pd.Grouper(freq="1h")).apply(
            lambda x: x.isnull().any()
        )
        print(hours_with_nans.shape)
        print(outputs.shape)
        print(hours_with_nans.tail())
        print(outputs.tail())
        outputs.loc[hours_with_nans] = np.nan

    # Save data
    outputs.to_hdf(folder + "outputs.h5", key="df")

    # Save outputs also for "country"-areas
    if area == "CE":
        folder = "../data/2020-2024/{}/".format("DE")
        if not os.path.exists(folder):
            os.makedirs(folder)
        outputs.to_hdf(folder + "outputs.h5", key="df")

        folder = "../data/2020-2024/{}/".format("CH")
        if not os.path.exists(folder):
            os.makedirs(folder)
        outputs.to_hdf(folder + "outputs.h5", key="df")

    if area == "Nordic":
        folder = "../data/2020-2024/{}/".format("SE")
        if not os.path.exists(folder):
            os.makedirs(folder)
        outputs.to_hdf(folder + "outputs.h5", key="df")
