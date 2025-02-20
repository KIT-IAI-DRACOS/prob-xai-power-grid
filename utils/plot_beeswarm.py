# Rewrite beeswarms function 
import matplotlib.pyplot as plt
import shap
import scipy
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as pl
from shap import Explanation
from shap.utils import safe_isinstance
from shap.utils._exceptions import DimensionError
from shap.plots import colors
from shap.plots._labels import labels
from shap.plots._utils import (
    convert_color,
    convert_ordering,
    get_sort_order,
    merge_nodes,
    sort_inds,
)

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

def get_shap_df(area, config, target, shap_type):
    X_test = pd.read_hdf(
                        data_folder.format(area, config.data_version, config.scaler_str)
                        + f"X_test_full{config.scaled}.h5")
    X_test.rename(columns=input_col_names, inplace=True)
    shap_vals = np.load(
            explain_folder.format(area, config.res_version, config.model_combination, target, config.explanations)
            + f"shap_values_{shap_type}.npy"
        )
    if len(shap_vals.shape) == 3 and shap_vals.shape[2] == 1:
        shap_vals = shap_vals.reshape(shap_vals.shape[0], shap_vals.shape[1])

    shap_vals = pd.DataFrame(
                    data=shap_vals, index=X_test.index, columns=X_test.columns
                )
    return shap_vals


def my_beeswarm(ax: plt.Axes, shap_values, labels: bool, max_display=10, order=Explanation.abs.mean(0),
             clustering=None, cluster_threshold=0.5, color=None,
             axis_color="#333333", alpha=1, show=True, log_scale=False,
             color_bar=True, s=16, plot_size="auto", color_bar_label=labels["FEATURE_VALUE"], clip_outliers = False):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).

    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further
        after it has been created, returning the current axis via plt.gca().

    color_bar : bool
        Whether to draw the color bar (legend).

    s : float
        What size to make the markers. For further information see `s` in ``matplotlib.pyplot.scatter``.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the
        number of features that are being displayed. Passing a single float will cause
        each row to be that many inches high. Passing a pair of floats will scale the
        plot by that number of inches. If ``None`` is passed, then the size of the
        current figure will be left unchanged.

    Examples
    --------
    See `beeswarm plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html>`_.

    """
    if not isinstance(shap_values, Explanation):
        emsg = (
            "The beeswarm plot requires an `Explanation` object as the "
            "`shap_values` argument."
        )
        raise TypeError(emsg)

    sv_shape = shap_values.shape
    if len(sv_shape) == 1:
        emsg = (
            "The beeswarm plot does not support plotting a single instance, please pass "
            "an explanation matrix with many instances!"
        )
        raise ValueError(emsg)
    elif len(sv_shape) > 2:
        emsg = (
            "The beeswarm plot does not support plotting explanations with instances that have more "
            "than one dimension!"
        )
        raise ValueError(emsg)

    shap_exp = shap_values
    values = np.copy(shap_exp.values)
    features = shap_exp.data
    if scipy.sparse.issparse(features):
        features = features.toarray()
    feature_names = shap_exp.feature_names

    if clip_outliers:
        # Apply quantile-based clipping to limit outliers
        shap_df = pd.DataFrame(values, columns=feature_names)
        for col in shap_df.columns:
            lower, upper = shap_df[col].quantile([0.001, 0.999])
            shap_df[col] = shap_df[col].clip(lower, upper)
        values = shap_df.values  # Update `values` after clipping


    order = convert_ordering(order, values)

    
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = values.shape[1]

    if features is not None:
        shape_msg = (
            "The shape of the shap_values matrix does not match the shape "
            "of the provided data matrix."
        )
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if log_scale:
        pl.xscale('symlog')

    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
        if partition_tree is not None and partition_tree.var(0).sum() == 0:
            partition_tree = partition_tree[0]
        else:
            partition_tree = None
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering

    if partition_tree is not None:
        if partition_tree.shape[1] != 4:
            emsg = (
                "The clustering provided by the Explanation object does not seem to "
                "be a partition tree (which is all shap.plots.bar supports)!"
            )
            raise ValueError(emsg)

   
    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()
    while True:
        feature_order = convert_ordering(order, Explanation(np.abs(values)))
        feature_order = np.arange(len(feature_order))
        break
        if partition_tree is not None:

            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values))

            # now relax the requirement to match the partition tree ordering for connections above cluster_threshold
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, cluster_threshold, feature_order)

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if max_display < len(feature_order) and dist[feature_order[max_display-1],feature_order[max_display-2]] <= cluster_threshold:
                #values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for i in range(len(values)):
                    values[:,ind1] += values[:,ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds)-1))
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features-1, len(values[0]))])
        values[:,feature_order[num_features-1]] = np.sum([values[:,feature_order[i]] for i in range(num_features-1, len(values[0]))], 0)

    # build our y-tick labels
    yticklabels = [feature_names[i] for i in feature_inds]
    if num_features < len(values[0]):
        yticklabels[-1] = "Sum of %d other features" % num_cut

    row_height = 0.4
    if False:
        if plot_size == "auto":
            pl.gcf().set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
        elif type(plot_size) in (list, tuple):
            pl.gcf().set_size_inches(plot_size[0], plot_size[1])
        elif plot_size is not None:
            pl.gcf().set_size_inches(8, min(len(feature_order), max_display) * plot_size + 1.5)
    ax.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]
        fvalues = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if fvalues is not None:
            fvalues = fvalues[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]: # check categorical feature
                colored_feature = False
            else:
                fvalues = np.array(fvalues, dtype=np.float64)  # make sure this can be numeric
        except Exception:
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if safe_isinstance(color, "matplotlib.colors.Colormap") and features is not None and colored_feature:
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax: # fixes rare numerical precision issues
                vmin = vmax

            if features.shape[0] != len(shaps):
                emsg = "Feature and SHAP matrices must have the same number of rows!"
                raise DimensionError(emsg)

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)
            ax.scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777",
                        s=s, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                        cmap=color, vmin=vmin, vmax=vmax, s=s,
                        c=cvals, alpha=alpha, linewidth=0,
                        zorder=3, rasterized=len(shaps) > 500)
        else:

            ax.scatter(shaps, pos + ys, s=s, alpha=alpha, linewidth=0, zorder=3,
                        color=color if colored_feature else "#777777", rasterized=len(shaps) > 500)
    

    # draw the color bar
    if False and safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)


    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    if labels:
        ax.set_yticks(range(len(feature_inds)), reversed(yticklabels), fontsize=12)
    else:
        ax.get_yaxis().set_visible(False)
    # ax.tick_params('y', length=20, width=0.5, which='major')
    # ax.tick_params('x', labelsize=11)
    ax.set_ylim(-1, len(feature_inds))
    #pl.xlabel(labels['VALUE'], fontsize=13)
    if show:
        pl.show()
    else:
        return pl.gca()


import os
import shap
import matplotlib.pyplot as plt

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
    if shap_type =="mean":
        plt.xlabel(r'SHAP value (impact on model output (mean prediction))', fontsize=12)
    elif shap_type=="std":
        plt.xlabel(r'SHAP value (impact on model output (uncertainty prediction))', fontsize=12)
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

# example use:
# plot_shap_values_beeswarm(config_ngb, area="CE", targ="f_rocof", shap_type="std", save=True)

def plot_shap_values_beeswarm_partition(config, area, targ, shap_type, save=False, cutoff=0.7):
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

    #shap_values_exp = shap.Explanation(shap_vals, feature_names=X_test.columns, data=X_test, clustering=clustering_reshaped)
    shap_values_exp = shap.Explanation(shap_vals, feature_names=X_test.columns, data=X_test)

    plt.figure(figsize=(10, 6))  # Beeswarm plot may require a bit more height than the bar plot
    print(shap_values_exp.shape)
    # Create the SHAP beeswarm plot
    shap.plots.beeswarm(shap_values_exp, show=False)

    # Set title and labels
    plt.title(r'\textbf{SHAP Beeswarm Plot for} \textit{' + targ + r'} \textbf{in ' + area + '}', fontsize=14)
    if shap_type =="mean":
        plt.xlabel(r'SHAP value (impact on model output (mean prediction))', fontsize=12)
    elif shap_type=="std":
        plt.xlabel(r'SHAP value (impact on model output (uncertainty prediction))', fontsize=12)
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

#plot_shap_values_beeswarm_partition(config_ngb, area="CE", targ="f_rocof", shap_type="std", save=False)

def plot_shap_beeswarm_for_clusters(config, area: str, target: str, shap_type: str, clustering, names_by_cluster: dict[int, list[str]], sorted_clusters: list[tuple]):
    shap_values = get_shap_df(area, config, target, shap_type)
    X_test = pd.read_hdf(
                    data_folder.format(area, config.data_version, config.scaler_str)
                    + f"X_test_full{config.scaled}.h5")
    X_test.rename(columns=input_col_names, inplace=True)

    # Create a beeswarm plot for each cluster in sorted order
    features_in_cluster = []
    for cluster in sorted_clusters[:8]:
        features_in_cluster += names_by_cluster[cluster]
   
    top_shap_vals = shap_values[features_in_cluster]
    cluster_shap_values = shap.Explanation(top_shap_vals.values, feature_names=features_in_cluster, data=X_test[features_in_cluster])
    print(cluster_shap_values)

    plt.figure(figsize= (8, 6))
    
    shap.plots.beeswarm(cluster_shap_values, show=False)
    #shap.summary_plot(cluster_shap_values.values, cluster_shap_values, plot_type="beeswarm", show=False)
    plt.title(f"SHAP Beeswarm Plot for Cluster {cluster} - {config.model_name}")
    plt.show()

#clustering, sorted_clusters, names_by_cluster = get_sorted_clusters(configs, area, shap_type, target, cutoff=0.7)
#plot_shap_beeswarm_for_clusters(config_tabnet, area, target, shap_type, clustering, names_by_cluster, sorted_clusters)

