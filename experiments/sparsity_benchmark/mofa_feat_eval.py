import json
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import mudata
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from addict import Dict
from tqdm.notebook import tqdm
from sklearn.metrics import r2_score

from cellij.core.synthetic import DataGenerator
from cellij.tools.evaluation import compute_factor_correlation
from cellij.utils import load_model, set_all_seeds
from cellij.core.models import MOFA


if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    CUDA = torch.cuda.is_available()
    torch.cuda.set_device(1)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    device = torch.device("cpu")


N_SHARED_FACTORS = 10
N_PARTIAL_FACTORS = 0
N_PRIVATE_FACTORS = 0
N_SAMPLES = 200
MISSINGS = 0.0
N_FACTORS_ESTIMATED = 20
OVERWRITE = False

perf_corr_factor_all = Dict()
perf_corr_factor = Dict()
perf_r2_x = Dict()
perf_time = Dict()
perf_w_activations_l1 = Dict()
perf_w_activations_l2 = Dict()
perf_w_activations_ve = Dict()

PATH_DGP = "/home/m015k/code/cellij/experiments/sparsity_benchmark/data/"
PATH_MODELS = "/data/m015k/data/cellij/benchmark/benchmark_v1_features/"


def compute_r2(y_true, y_predicted):
    sse = np.sum((y_true - y_predicted) ** 2)
    # tse = np.sum((y_true - np.mean(y_true))**2)
    tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score, sse, tse


for seed in [0, 1]:  # 2, 3, 4
    set_all_seeds(seed)

    for grid_features in tqdm(
        [50, 100, 200, 400, 800, 1000, 2000, 5000]
    ):  # 50, 100, 10000
        n_samples = N_SAMPLES
        n_features = [grid_features, grid_features, grid_features]
        n_views = len(n_features)
        likelihoods = ["Normal" for _ in range(n_views)]

        n_fully_shared_factors = N_SHARED_FACTORS
        n_partially_shared_factors = N_PARTIAL_FACTORS
        n_private_factors = N_PRIVATE_FACTORS
        n_covariates = 0

        if not (
            Path(PATH_DGP)
            .joinpath(
                f"dgp_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
            )
            .exists()
        ):
            print("Neeeeein....")
        else:
            print(f"Loading data from {PATH_DGP}...")
            mdata = mudata.read(
                Path(PATH_DGP).joinpath(
                    f"dgp_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                )
            )
            y = np.concatenate([mdata[m].X for m in mdata.mod], axis=1)

        for lr in [0.1, 0.01, 0.001]:  # , 0.0001
            for sparsity_prior, prior_params in [
                (None, {}),
                ("Lasso", {"lasso_scale": 0.1}),
                ("Horseshoe", {"tau_scale": 1.0, "lambda_scale": 1.0}),
                ("Horseshoe", {"tau_scale": 0.1, "lambda_scale": 1.0}),
                (
                    "Horseshoe",
                    {"tau_scale": 0.1, "lambda_scale": 1.0, "delta_tau": True},
                ),
                (
                    "Horseshoe",
                    {"tau_scale": 1.0, "lambda_scale": 1.0, "regularized": True},
                ),
                ("HorseshoePlus", {"tau_const": 0.1, "eta_scale": 1.0}),
                ("SpikeNSlab", {"relaxed_bernoulli": True, "temperature": 0.1}),
                ("SpikeNSlab", {"relaxed_bernoulli": True, "temperature": 0.01}),
                ("SpikeNSlab", {"relaxed_bernoulli": False}),
                (
                    "SpikeNSlabLasso",
                    {
                        "lambda_spike": 20.0,
                        "lambda_slab": 1.0,
                        "relaxed_bernoulli": True,
                        "temperature": 0.1,
                    },
                ),
                (
                    "SpikeNSlabLasso",
                    {
                        "lambda_spike": 20.0,
                        "lambda_slab": 0.01,
                        "relaxed_bernoulli": True,
                        "temperature": 0.1,
                    },
                ),
            ]:
                print(
                    f"{seed} | {grid_features} | {lr} | {sparsity_prior} | {prior_params}"
                )
                # Combine all parameters used for the prior into a string
                # This allows to train model with the same prior but different parameters
                s_params = (
                    "_".join([f"{k}={v}" for k, v in prior_params.items()])
                    if prior_params
                    else "None"
                )
                filename = Path(PATH_MODELS).joinpath(
                    f"model_v1_features_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{sparsity_prior}_{N_FACTORS_ESTIMATED}_{lr}_{seed}_{s_params}.pkl"
                )

                if Path(filename).exists():
                    print(f"Loading {filename}")
                    model = load_model(str(filename))

                    # Load json file
                    with open(
                        str(filename).replace(".pkl", "_perf_timer.json"), "r"
                    ) as fp:
                        perf_timer = json.load(fp)

                else:
                    print("Model not found")
                    continue

                if sparsity_prior in ["SpikeNSlabLasso"]:
                    z_hat = model._guide.median()["z"]
                    w_hat = torch.cat(
                        [
                            (
                                (1 - model._guide.median()[f"lambda_view_{m}"])
                                * model._guide.median()[f"lambda_spike_view_{m}"]
                                + model._guide.median()[f"lambda_view_{m}"]
                                * model._guide.median()[f"lambda_slab_view_{m}"]
                            ).squeeze()
                            for m in range(n_views)
                        ],
                        dim=1,
                    )

                elif hasattr(model._guide, "mode"):
                    z_hat = model._guide.mode()["z"]
                    d = model._guide.mode()
                    if d.get("w_view_0") is not None:
                        query_key = "w_view_"
                    else:
                        query_key = "w_feature_group_"

                    w_hat = torch.cat(
                        [
                            model._guide.mode()[f"{query_key}{m}"].squeeze()
                            for m in range(n_views)
                        ],
                        dim=1,
                    )

                else:
                    z_hat = model._guide.median()["z"]
                    d = model._guide.median()
                    if d.get("w_view_0") is not None:
                        query_key = "w_view_"
                    else:
                        query_key = "w_feature_group_"

                    w_hat = torch.cat(
                        [
                            model._guide.median()[f"{query_key}{m}"].squeeze()
                            for m in range(n_views)
                        ],
                        dim=1,
                    )

                temp_model = MOFA(n_factors=100, sparsity_prior="Horseshoe")
                temp_model.add_data(data=mdata, na_strategy=None)
                x_true = temp_model._data.values

                z_hat = z_hat.detach().cpu().numpy().squeeze()
                w_hat = w_hat.detach().cpu().numpy().squeeze()

                if sparsity_prior is None:
                    sparsity_prior = "Normal"
                else:
                    sparsity_prior = sparsity_prior + "_" + s_params

                legend_names = {
                    "Normal": "Normal",
                    "Lasso_lasso_scale=0.1": "Laplace(0,0.1)",
                    "Horseshoe_tau_scale=0.1_lambda_scale=1.0": "HS (0.1, 1)",
                    "Horseshoe_tau_scale=0.1_lambda_scale=1.0_delta_tau=True": "HS Const. Tau (0.1, 1)",
                    "Horseshoe_tau_scale=1.0_lambda_scale=1.0": "HS (1.0, 1)",
                    "Horseshoe_tau_scale=1.0_lambda_scale=1.0_regularized=True": "HS Reg. (1, 1)",
                    "HorseshoePlus_tau_const=0.1_eta_scale=1.0": "HS+ (0.1, 1)",
                    "SpikeNSlab_relaxed_bernoulli=False": "SnS CB (20, 1)",
                    "SpikeNSlab_relaxed_bernoulli=True_temperature=0.1": "SnS RB (0.1)",
                    "SpikeNSlab_relaxed_bernoulli=True_temperature=0.01": "SnS RB (0.01)",
                    "SpikeNSlabLasso_lambda_spike=20.0_lambda_slab=1.0_relaxed_bernoulli=True_temperature=0.1": "SnS-L RB (20, 1.0, 0.1)",
                    "SpikeNSlabLasso_lambda_spike=20.0_lambda_slab=0.01_relaxed_bernoulli=True_temperature=0.1": "SnS-L RB (20, 0.01, 0.1)",
                }
                if sparsity_prior in legend_names:
                    sparsity_prior = legend_names[sparsity_prior]

                # Reconstruction
                x_hat = np.matmul(z_hat, w_hat)
                perf_r2_x[seed][grid_features][lr][sparsity_prior] = r2_score(
                    x_true, x_hat
                )

                # R2 of factors
                w_hats = np.split(w_hat, 3, axis=1)
                for idx, m in enumerate(mdata.mod):
                    w_true = mdata.mod[m].varm["w"]
                    avg, all_k, _, _ = compute_factor_correlation(w_true, w_hats[idx].T)
                    perf_corr_factor[seed][grid_features][lr][sparsity_prior] = avg
                    perf_corr_factor_all[seed][grid_features][lr][
                        sparsity_prior
                    ] = all_k  # TODO: Plot as well

                # Timer
                perf_time[seed][grid_features][lr][sparsity_prior] = perf_timer

                # W activation
                # Split w_hat into three equally sized parts along axis=1
                w_hats = np.split(w_hat, 3, axis=1)
                x_true_splitted = np.split(x_true, 3, axis=1)
                for m, name in enumerate(mdata.mod):
                    for k in range(w_hats[m].shape[0]):
                        perf_w_activations_l1[seed][grid_features][lr][sparsity_prior][
                            m
                        ][k] = np.sum(np.abs(w_hats[m][k]))
                        perf_w_activations_l2[seed][grid_features][lr][sparsity_prior][
                            m
                        ][k] = np.sum(w_hats[m][k] ** 2)
                        x_hat_from_single_k = (
                            w_hats[m][k][:, None] @ z_hat[:, k][None, :]
                        )
                        x_hat_from_single_k = x_hat_from_single_k.T
                        perf_w_activations_ve[seed][grid_features][lr][sparsity_prior][
                            m
                        ][k] = (
                            compute_r2(x_true_splitted[m], x_hat_from_single_k)[0]
                            / x_true_splitted[m].size
                        )


#
# Plots
#


def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df


# Prepare dataframes for plotting
df_r2 = nested_dict_to_df(perf_corr_factor.to_dict()).reset_index()
for k, col_name in enumerate(["seed", "grid_features", "lr"]):
    # Rename kth columns
    df_r2 = df_r2.rename(columns={f"level_{k}": col_name})

df_r2 = df_r2.melt(
    id_vars=["seed", "grid_features", "lr"], var_name="sparsity_prior", value_name="r2"
)

df_time = nested_dict_to_df(perf_time.to_dict()).reset_index()
# Assign columns names from 1 to n
df_time.columns = [f"{x}" for x in range(1, len(df_time.columns) + 1)]
df_time_idx = df_time.iloc[:, :4]
# Coalesce all rows
df_time_idx = df_time_idx.assign(time=np.nansum(df_time.iloc[:, 4:].to_numpy(), axis=1))
df_time_idx.columns = ["seed", "grid_features", "lr", "sparsity_prior", "time"]


# Sort all rows independelty in descending order
# Define a function to sort each row
def sort_row_normalized(row):
    return pd.Series(sorted(row, reverse=True), index=row.index) / np.abs(max(row))


def sort_row(row):
    return pd.Series(sorted(row, reverse=True), index=row.index)


# Apply the function to each row using apply with axis=1
df_w_act_l1 = nested_dict_to_df(perf_w_activations_l1.to_dict()).reset_index()
df_w_act_l1_idx = df_w_act_l1.iloc[:, :5]
df_w_act_l1 = df_w_act_l1.iloc[:, 5:]
df_w_act_l1 = df_w_act_l1.apply(sort_row_normalized, axis=1)
df_w_act_l1 = pd.concat([df_w_act_l1_idx, df_w_act_l1], axis=1)
df_w_act_l1.columns = ["seed", "grid_features", "lr", "sparsity_prior", "view"] + [
    f"factor_{x}" for x in range(1, N_FACTORS_ESTIMATED + 1)
]

df_w_act_l2 = nested_dict_to_df(perf_w_activations_l2.to_dict()).reset_index()
df_w_act_l2_idx = df_w_act_l2.iloc[:, :5]
df_w_act_l2 = df_w_act_l2.iloc[:, 5:]
df_w_act_l2 = df_w_act_l2.apply(sort_row_normalized, axis=1)
df_w_act_l2 = pd.concat([df_w_act_l2_idx, df_w_act_l2], axis=1)
df_w_act_l2.columns = ["seed", "grid_features", "lr", "sparsity_prior", "view"] + [
    f"factor_{x}" for x in range(1, N_FACTORS_ESTIMATED + 1)
]


df_w_act_ve = nested_dict_to_df(perf_w_activations_ve.to_dict()).reset_index()
df_w_act_ve_idx = df_w_act_ve.iloc[:, :5]
df_w_act_ve = df_w_act_ve.iloc[:, 5:]
df_w_act_ve = df_w_act_ve.apply(sort_row, axis=1)
df_w_act_ve = pd.concat([df_w_act_ve_idx, df_w_act_ve], axis=1)
df_w_act_ve.columns = ["seed", "grid_features", "lr", "sparsity_prior", "view"] + [
    f"factor_{x}" for x in range(1, N_FACTORS_ESTIMATED + 1)
]


sns.set_theme(style="whitegrid")
sns.set_context("paper")
sns.set_palette(sns.color_palette("colorblind", 9))
plt.rcParams["figure.constrained_layout.use"] = False
plt.rcParams["figure.autolayout"] = False
# plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.015

#
# Plot R2 of average factor reconstruction with respect to features
#
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

df_r2 = df_r2.sort_values(by=["sparsity_prior"])
fig, ax = plt.subplots(1, 1, figsize=(6.75, 5))
g = sns.boxplot(
    data=df_r2,
    x="grid_features",
    y="r2",
    hue="sparsity_prior",
).set(xlabel="Number of Features", ylabel="Correlation of Factors")
# Place location below plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.grid(True)
# Set y range to 0 to 1
plt.ylim(0.2, 1.05)
# Save plot as pdf and png
plt.tight_layout()
plt.savefig("plots/r2_features.pdf")
plt.savefig("plots/r2_features.png")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6.75, 5))
for feats in df_r2["grid_features"].unique():
    df_r2_tmp = df_r2[df_r2["grid_features"] == feats]
    g = sns.boxplot(
        data=df_r2_tmp,
        x="sparsity_prior",
        y="r2",
    ).set(xlabel="Sparsity Priors", ylabel="Correlation of Factors")
    plt.title(f"Number of Features = {feats}")
    plt.grid(True)
    # Set y range to 0 to 1
    plt.ylim(0.25, 1.05)
    # Save plot as pdf and png
    plt.tight_layout()
    # Rotate x labels by 90 degrees
    plt.xticks(rotation=90)
    plt.savefig(f"plots/r2_general_{feats}.pdf")
    plt.savefig(f"plots/r2_general_{feats}.png")
    plt.show()


#
# Plot Runtime with respect to features
#
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(6.75, 5))
g = sns.boxplot(
    data=df_time_idx,
    x="grid_features",
    y="time",
    hue="sparsity_prior",
).set(xlabel="Number of Features", ylabel="Time [s]")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.tight_layout()
plt.grid(True)
plt.show()


#
# Plot #Active columns reconstruction with respect to features
#
BASE_DATA = df_w_act_l2

# plt.style.use('default')
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
# from tueplots import bundles

# def _from_base_in(*, base_width_in, rel_width, height_to_width_ratio, nrows, ncols):
#     width_in = base_width_in * rel_width
#     subplot_width_in = width_in / ncols
#     subplot_height_in = height_to_width_ratio * subplot_width_in
#     height_in = subplot_height_in * nrows
#     return width_in, height_in

# figsize = _from_base_in(
#     base_width_in=6.75,
#     rel_width=1.0,
#     height_to_width_ratio=(5.0**0.5 - 1.0) / 2.0,
#     nrows=1,
#     ncols=2,
# )

# plt.rcParams["figure.family"] = "sans-serif"
# plt.rcParams["figure.figsize"] = figsize
# bundles.icml2022()

# BASE_DATA_melt = BASE_DATA.melt(
#     id_vars=["seed", "grid_features", "lr", "sparsity_prior", "view"],
#     var_name="factor",
#     value_name="L2 Norm",
# )
# # Replace factor names with numbers and convert to int
# BASE_DATA_melt["factor"] = (
#     BASE_DATA_melt["factor"].str.replace("factor_", "").astype(int)
# )
# # Plot L1 norms of W for each sparsity prior separately
# fig, ax = plt.subplots(
#     BASE_DATA_melt["sparsity_prior"].nunique(),
#     1,
#     figsize=(6.75, 20),
#     sharex=False,
#     sharey=False,
#     tight_layout=False,
# )
# for idx, sparsity_prior in enumerate(BASE_DATA_melt["sparsity_prior"].unique()):
#     # Draw a vertical line at N_FACTORS_TRUE
#     ax[idx].axvline(x=N_SHARED_FACTORS + 0.5, color="gray", linestyle="--", linewidth=2)

#     BASE_DATA_melt_sp = BASE_DATA_melt[
#         BASE_DATA_melt["sparsity_prior"] == sparsity_prior
#     ]
#     # Make grid features a categorical variable using .loc
#     BASE_DATA_melt_sp.loc[:, "grid_features"] = BASE_DATA_melt_sp[
#         "grid_features"
#     ].astype("str")
#     # Use colorpalette with 3 colors
#     sns.lineplot(
#         data=BASE_DATA_melt_sp,
#         x="factor",
#         y="L2 Norm",
#         hue="grid_features",
#         ax=ax[idx],
#         legend=False if idx < BASE_DATA_melt["sparsity_prior"].nunique() - 1 else True,
#     )
#     # Add title to each subplot with name of sparse prior
#     ax[idx].set_title(f"Sparsity Prior: {sparsity_prior}")
#     # Show only integer ticks on x axis from 1 to N_FACTORS_ESTIMATED
#     ax[idx].set_xticks(range(1, N_FACTORS_ESTIMATED + 1))

# # Set x and y labels
# plt.xlabel("Factor")
# plt.ylabel("L2 Norm of Factor")
# # Place location below plot
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4)
# plt.grid(True)
# # do not show ax[5,1] since it is empty
# plt.show()


BASE_DATA_melt = BASE_DATA.melt(
    id_vars=["seed", "grid_features", "lr", "sparsity_prior", "view"],
    var_name="factor",
    value_name="L2 Norm",
)
# Replace factor names with numbers and convert to int
BASE_DATA_melt["factor"] = (
    BASE_DATA_melt["factor"].str.replace("factor_", "").astype(int)
)
# Plot L1 norms of W for each sparsity prior separately
# Create a figure with two columns
Ns = BASE_DATA_melt["sparsity_prior"].nunique()
nrows = int(np.ceil(BASE_DATA_melt["sparsity_prior"].nunique() / 3))
fig, ax = plt.subplots(
    nrows,
    3,
    figsize=(6.75, 12.5),
    sharex=False,
    sharey=True,
    tight_layout=False,
)
for idx, sparsity_prior in enumerate(BASE_DATA_melt["sparsity_prior"].unique()):
    # Draw a vertical line at N_FACTORS_TRUE
    ax[idx - nrows * (idx // nrows), idx // nrows].axvline(
        x=N_SHARED_FACTORS + 0.5, color="gray", linestyle="--", linewidth=2
    )

    BASE_DATA_melt_sp = BASE_DATA_melt[
        BASE_DATA_melt["sparsity_prior"] == sparsity_prior
    ]
    # Make grid features a categorical variable using .loc
    BASE_DATA_melt_sp.loc[:, "grid_features"] = BASE_DATA_melt_sp[
        "grid_features"
    ].astype("str")
    # Use colorpalette with 3 colors
    sns.lineplot(
        data=BASE_DATA_melt_sp,
        x="factor",
        y="L2 Norm",
        hue="grid_features",
        ax=ax[idx - nrows * (idx // nrows), idx // nrows],
        legend=False if idx != 7 else True,
    )
    # Add title to each subplot with name of sparse prior
    ax[idx - nrows * (idx // nrows), idx // nrows].set_title(f"{sparsity_prior}")
    # Show only integer ticks on x axis from 1 to N_FACTORS_ESTIMATED
    ax[idx - nrows * (idx // nrows), idx // nrows].set_xticks(
        range(1, N_FACTORS_ESTIMATED + 1)
    )
    # Set fontsize of x ticks
    ax[idx - nrows * (idx // nrows), idx // nrows].tick_params(axis="x", labelsize=6)
    # Set x and y labels
    ax[idx - nrows * (idx // nrows), idx // nrows].set_xlabel("Factor")
    ax[idx - nrows * (idx // nrows), idx // nrows].set_ylabel("L2 Norm of Factor")

# Place location below plot
ax[3, 1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.3),
    ncol=2,
    title="Features",
    fancybox=True,
    shadow=False,
    fontsize=10,
)
plt.grid(True)
plt.tight_layout()
# ax[3, 2].axis("off")
plt.savefig("plots/l2.pdf")
plt.savefig("plots/l2.png")
# plt.subplots_adjust(wspace=0, hspace=0)
# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)
plt.show()
