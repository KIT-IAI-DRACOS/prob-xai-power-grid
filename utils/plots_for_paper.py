import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm



data_folder = "../../data/2020-2024/{}/version_{}/{}/"
explain_folder = "../../explanations/{}/version_{}_{}/target_{}/"
input_cols = [
    "gen_other",
    "gen_solar",
    "gen_wind_on",
    "gen_waste",
    "gen_nuclear",
    "gen_biomass",
    "gen_gas",
    "gen_run_off_hydro",
    "gen_oil",
    "gen_pumped_hydro",
    "gen_other_renew",
    "gen_reservoir_hydro",
    "gen_hard_coal",
    "gen_wind_off",
    "gen_geothermal",
    "gen_lignite",
    "load",
    "gen_coal_gas",
    "total_gen",
    "synchronous_gen",
    "load_ramp",
    "total_gen_ramp",
    "other_ramp",
    "solar_ramp",
    "wind_on_ramp",
    "waste_ramp",
    "nuclear_ramp",
    "biomass_ramp",
    "gas_ramp",
    "run_off_hydro_ramp",
    "oil_ramp",
    "pumped_hydro_ramp",
    "other_renew_ramp",
    "reservoir_hydro_ramp",
    "hard_coal_ramp",
    "wind_off_ramp",
    "geothermal_ramp",
    "lignite_ramp",
    "coal_gas_ramp",
    "forecast_error_wind_on",
    "forecast_error_wind_off",
    "forecast_error_solar",
    "forecast_error_total_gen",
    "forecast_error_load",
    "forecast_error_load_ramp",
    "forecast_error_total_gen_ramp",
    "forecast_error_wind_off_ramp",
    "forecast_error_wind_on_ramp",
    "forecast_error_solar_ramp",
    "solar_day_ahead",
    "wind_on_day_ahead",
    "scheduled_gen_total",
    "prices_day_ahead",
    "load_day_ahead",
    "wind_off_day_ahead",
    "month",
    "weekday",
    "hour",
    "load_ramp_day_ahead",
    "total_gen_ramp_day_ahead",
    "wind_off_ramp_day_ahead",
    "wind_on_ramp_day_ahead",
    "solar_ramp_day_ahead",
    "price_ramp_day_ahead",
    "gen_fossil_peat",
    "fossil_peat_ramp",
    "residual",
]


input_col_names = [
    "Generation other",
    "Solar generation",
    "Onshore wind generation",
    "Waste generation",
    "Nuclear generation",
    "Biomass generation",
    "Gas generation",
    "Run-off-river hydro generation",
    "Oil generation",
    "Pumped hydro generation",
    "Other renewable generation",
    "Reservoir hydro generation",
    "Hard coal generation",
    "Wind offshore generation",
    "Geothermal generation",
    "Lignite generation",
    "Load",
    "Coal gas generation",
    "Total generation",
    "Synchronous generation",
    "Load ramp",
    "Total generation ramp",
    "Other ramp",
    "Solar ramp",
    "Onshore wind ramp",
    "Waste ramp",
    "Nuclear ramp",
    "Biomass ramp",
    "Gas ramp",
    "Run-off-river hydro ramp",
    "Oil ramp",
    "Pumped hydro ramp",
    "Other renewable ramp",
    "Reservoir hydro ramp",
    "Hard coal ramp",
    "Offshore wind ramp",
    "geothermal_ramp",
    "Lignite ramp",
    "Coal gas ramp",
    "Forecast error onshore wind",
    "Forecast error offshore wind",
    "Forecast error solar",
    "Forecast error total generation",
    "Forecast error load",
    "Forecast error load ramp",
    "Forecast error generation ramp",
    "Forecast error offshore wind ramp",
    "Forecast error onshore wind ramp",
    "Forecast error solar ramp",
    "Solar day-ahead",
    "Onshore wind day-ahead",
    "Scheduled generation",
    "Prices day-ahead",
    "Load day-ahead",
    "Offshore wind day-ahead",
    "Month",
    "Weekday",
    "Hour",
    "Load ramp day-ahead",
    "Generation ramp day-ahead",
    "Offshore wind ramp day-ahead",
    "Onshore wind ramp day-ahead",
    "Solar ramp day-ahead",
    "Price ramp day-ahead",
    "Fossil peat generation",
    "Fossil peat ramp",
    "Residual",
]

input_col_names = dict(zip(input_cols, input_col_names))
#input_col_names_units = dict(zip(input_cols, input_col_names_units))
#input_col_names_units_general = dict(zip(input_cols, input_col_names_units_general))

input_rescale_factors = pd.Series(index=input_cols, data=1 / 1000)
input_rescale_factors.loc[
    ["weekday", "hour", "month", "prices_day_ahead", "price_ramp_day_ahead"]
] = 1


def plot_shap_values_bar(config, area, targ, shap_type, save=False):
    """
    Plot SHAP values with an option to save the figure.

    Parameters:
    - config: Configuration object with necessary parameters.
    - area: The area for which SHAP values are being plotted.
    - targ: The target feature for which SHAP values are being plotted.
    - shap_type: Type of SHAP values ('mean' or 'std', etc.).
    - save: Boolean flag indicating whether to save the figure. Default is False.
    """

    X_test = pd.read_hdf(
                    data_folder.format(area, config.data_version, config.scaler_str)
                    + f"X_test_full{config.scaled}.h5")
    X_test.rename(columns=input_col_names, inplace=True)

    shap_vals = np.load(
            explain_folder.format(area, config.res_version, config.model_combination, targ, config.explanations)
            + f"shap_values_{shap_type}.npy"
        )

    shap_values_exp = shap.Explanation(shap_vals, feature_names=X_test.columns, data=X_test)

    plt.figure(figsize=(10, 4))

    shap.plots.bar(shap_values_exp, max_display=11, show=False)

    """
    # Set KIT colors:
    ax = plt.gca()

    for bar in ax.patches:
        bar.set_color('#009682')
    """

    plt.title(r'\textbf{SHAP Feature Importance for} \textit{' + targ + r'} \textbf{in ' + area + '}', fontsize=14)
    plt.xlabel(r'Mean $|$SHAP value$|$ (impact on model output)', fontsize=12)

    plt.tight_layout()
    # Save or show the plot
    if save:
        save_dir = explain_folder.format(area, config.res_version, config.model_combination, targ, config.explanations) + f"plots/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + f"shap_bar_plot_{shap_type}.pdf"
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        print(f"Figure saved at {save_path}")
        plt.show()
    else:
        plt.show() 

#example:
#plot_shap_values_bar(config_xgb, area="Nordic", targ="f_integral", shap_type="mean", save=True)



def plot_shap_values_beeswarm(config, area, targ, shap_type, save=False):
    """
    Plot SHAP beeswarm plot with an option to save the figure.

    Parameters:
    - config: Configuration object with necessary parameters.
    - area: The area for which SHAP values are being plotted.
    - targ: The target feature for which SHAP values are being plotted.
    - shap_type: Type of SHAP values ('mean' or 'std', etc.).
    - save: Boolean flag indicating whether to save the figure. Default is False.
    """

    # Load the test data and SHAP values
    X_test = pd.read_hdf(
                    data_folder.format(area, config.data_version, config.scaler_str)
                    + f"X_test_full{config.scaled}.h5")
    X_test.rename(columns=input_col_names, inplace=True)

    shap_vals = np.load(
            explain_folder.format(area, config.res_version, config.model_combination, targ, config.explanations)
            + f"shap_values_{shap_type}.npy"
        )

    # Create a SHAP explanation object
    shap_values_exp = shap.Explanation(shap_vals, feature_names=X_test.columns, data=X_test)

    plt.figure(figsize=(10, 6))  # Beeswarm plot may require a bit more height than the bar plot

    # Create the SHAP beeswarm plot
    shap.plots.beeswarm(shap_values_exp, show=False)

    # Set title and labels
    plt.title(r'\textbf{SHAP Beeswarm Plot for} \textit{' + targ + r'} \textbf{in ' + area + '}', fontsize=14)
    plt.xlabel(r'SHAP value (impact on model output)', fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save or show the plot
    if save:
        save_dir = explain_folder.format(area, config.res_version, config.model_combination, targ, config.explanations) + f"plots/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + f"shap_beeswarm_plot_{shap_type}.pdf"
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        print(f"Figure saved at {save_path}")
        plt.show()
    else:
        plt.show()


def line_plot_daily_profile(area: str, target: str, num_samples: int, save: bool):
    """
    Plot the daily profile in comparison to num_samples random days
    """
    version_folder = f"../../data/2020-2024/{area}/version_2024-05-19/"
    figure_path = "../../results/figures/"
    X_train = pd.read_hdf(version_folder + "X_train_full.h5")
    X_test = pd.read_hdf(version_folder + "X_test_full.h5")
    X_full = pd.concat([X_train, X_test], axis=0)
    y_train = pd.read_hdf(
            version_folder.format(area)
            + f"y_train.h5"
        ).loc[:, target]
    y_test = pd.read_hdf(
                version_folder.format(area)
                + f"y_test.h5"
            ).loc[:, target]
    y_full = pd.concat([y_train, y_test], axis=0)
    daily_profile_full = y_full.groupby(X_full.index.time).mean()

    random_days = y_full.resample('D').mean().sample(num_samples).index

    random_days_data = y_full[y_full.index.floor('D').isin(random_days)]

    plt.figure(figsize=(12, 6))
    for i, (day, data) in enumerate(random_days_data.groupby(random_days_data.index.floor('D'))):
        data = data.sort_index(key=lambda x: x.hour)
        label = f"Day: {day.date()}" if i == 0 else None
        plt.plot(data.index.hour, data.values, label=label, alpha=0.45)


    if isinstance(daily_profile_full.index[0], datetime.time):
        daily_profile_full.index = [t.hour for t in daily_profile_full.index]
    plt.plot(daily_profile_full.index, daily_profile_full.values, color='black', linewidth=2, label='Daily Profile', linestyle='--')


    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel(f'{target} value', fontsize=12)
    plt.legend()

    if save:
        save_path = figure_path + "daily_profile.pdf"
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        print(f"Figure saved at {save_path}")
        plt.show()
    else:
        plt.show() 

#line_plot_daily_profile("CE","f_rocof", 300, save=True)

def line_plot_daily_profile_std(area: str, target: str, num_samples: int, save: bool):
    """
    Plot the daily profile (standard deviation) in comparison to num_samples random days
    """
    version_folder = f"../../data/2020-2024/{area}/version_2024-05-19/"
    figure_path = "../../results/figures/"
    X_train = pd.read_hdf(version_folder + "X_train_full.h5")
    X_test = pd.read_hdf(version_folder + "X_test_full.h5")
    X_full = pd.concat([X_train, X_test], axis=0)
    y_train = pd.read_hdf(
            version_folder.format(area)
            + f"y_train.h5"
        ).loc[:, target]
    y_test = pd.read_hdf(
                version_folder.format(area)
                + f"y_test.h5"
            ).loc[:, target]
    y_full = pd.concat([y_train, y_test], axis=0)
    daily_profile_full = y_full.groupby(X_full.index.time).mean()
    daily_profile_std = y_full.groupby(X_full.index.time).std()

    random_days = y_full.resample('D').mean().sample(num_samples).index

    random_days_data = y_full[y_full.index.floor('D').isin(random_days)]

    plt.figure(figsize=(12, 6))
    for i, (day, data) in enumerate(random_days_data.groupby(random_days_data.index.floor('D'))):
        data = data.sort_index(key=lambda x: x.hour)
        label = f"Day: {day.date()}" if i == 0 else None
        plt.plot(data.index.hour, data.values, label=label, alpha=0.35)


    if isinstance(daily_profile_full.index[0], datetime.time):
        daily_profile_full.index = [t.hour for t in daily_profile_full.index]
    plt.plot(daily_profile_full.index, daily_profile_full.values, color='black', linewidth=2, label='Daily Profile', linestyle='--')

    plt.fill_between(daily_profile_full.index, 
                 daily_profile_full.values - daily_profile_std.values, 
                 daily_profile_full.values + daily_profile_std.values, 
                 color='grey', alpha=0.5, label='Daily Profile ± Std Dev')
    plt.plot(daily_profile_full.index, daily_profile_full.values + daily_profile_std.values, color='green',linestyle='--', linewidth=2)
    plt.plot(daily_profile_full.index, daily_profile_full.values - daily_profile_std.values, color='green',linestyle='--', linewidth=2, label="Daily Profile: Upper and lower bounds")


    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel(f'{target} value', fontsize=12)
    plt.legend()

    if save:
        save_path = figure_path + "daily_profile_std2.pdf"
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        print(f"Figure saved at {save_path}")
        plt.show()
    else:
        plt.show() 


def plot_crps():
    x = np.linspace(-3, 3, 1000)  
    observed_value = 0.5  

    mu, sigma = 0, 1  
    forecast_cdf = norm.cdf(x, loc=mu, scale=sigma)

    observed_cdf = np.where(x >= observed_value, 1, 0)

    plt.figure(figsize=(8, 6))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(x, forecast_cdf, label=r'Predicted CDF', color='blue')
    plt.step(x, observed_cdf, label=r'Observed CDF', where='post', color='orange')

    plt.fill_between(x, forecast_cdf, observed_cdf, color='gray', alpha=0.3, label=r'CRPS Area')

    plt.xlabel(r'Possible Outcomes', fontsize=14)
    plt.ylabel(r'Cumulative Distribution Function (CDF)', fontsize=14)

    plt.legend(fontsize=12)
    figure_path = (
        "../../results/figures/"
    )
    plt.savefig(figure_path+ 'crps_illustration.pdf', format='pdf')
    plt.show()


import scipy.cluster.hierarchy as sch

def plot_correlation_clustering(area, data_version="2024-05-19", input_cols=None, save=False):
    version_folder = f"../data/2020-2024/{area}/version_{data_version}/yeo_johnson/"
    X_train = pd.read_hdf(version_folder + "X_train_full_scaled.h5")
    
    figure_path = (
        f"../../probabilistic-XAI-for-grid-frequency-stability/results/figures/correlation/{area}/"
    )
    
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    
    correlation_matrix = X_train.corr(method='pearson')
    correlation_matrix = np.abs(np.corrcoef(correlation_matrix))
    dist_matrix = 1 - correlation_matrix
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    clustering = sch.linkage(dist_matrix, method="complete", optimal_ordering=True)
    
    if input_cols:
        labels = [input_cols.get(col, col) for col in X_train.columns]
    else:
        labels = X_train.columns
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    plt.figure(figsize=(15, 7))
    plt.title(r"\textbf{Dendrogram of Correlation Clustering of " + area + ' Features}', fontsize=14)
    dend = sch.dendrogram(clustering, labels=labels)
    
    plt.xticks(rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=12)
    plt.ylabel(r'\textbf{Correlation Distance}', fontsize=12)
    
    if save:
        plt.savefig(figure_path + "pearson_corr.pdf", bbox_inches='tight')
    plt.show()
    
    return clustering

