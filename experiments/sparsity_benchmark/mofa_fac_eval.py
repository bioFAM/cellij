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

from cellij.core.models import MOFA
from cellij.core.synthetic import DataGenerator
from cellij.tools.evaluation import compute_factor_correlation
from cellij.utils import load_model, set_all_seeds


if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    CUDA = torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    device = torch.device("cpu")

N_SHARED_FACTORS = 10
N_PARTIAL_FACTORS = 0
N_PRIVATE_FACTORS = 0
# N_SAMPLES = 200
MISSINGS = 0.0
# N_FACTORS_ESTIMATED = 20
OVERWRITE = False

PATH_DGP = "/home/m015k/code/cellij/experiments/sparsity_benchmark/data/"
PATH_MODELS = "/data/m015k/data/cellij/benchmark/benchmark_v1_factors/"

perf_w_activations_l1 = Dict()
perf_w_activations_l2 = Dict()
perf_w_activations_ve = Dict()

def compute_r2(y_true, y_predicted):
    sse = np.sum((y_true - y_predicted) ** 2)
    # tse = np.sum((y_true - np.mean(y_true))**2)
    tse = (len(y_true) - 1) * np.var(y_true, ddof=1)
    r2_score = 1 - (sse / tse)
    return r2_score, sse, tse

for seed in [0, 1, 2]:  #  2, 3, 4
    set_all_seeds(seed)

    for N_SAMPLES in [100, 200, 500]:
        for grid_features in [300, 1000, 5000]:
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
                pass
            else:
                print(f"Loading data from {PATH_DGP}...")
                mdata = mudata.read(
                    Path(PATH_DGP).joinpath(
                        f"dgp_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                    )
                )

            for lr in [0.1, 0.01, 0.001]:  # , 0.0001
                for N_FACTORS_ESTIMATED in [50, 60]:  #  5, 10, 15, 20, 25, 40, 50, 60
                    for sparsity_prior, prior_params in [
                        (None, {}),
                        ("SpikeNSlab", {"relaxed_bernoulli": True, "temperature": 0.1}),
                        ("SpikeNSlab", {"relaxed_bernoulli": False}),
                        ("Lasso", {"lasso_scale": 0.1}),
                        ("Horseshoe", {"tau_scale": 1.0, "lambda_scale": 1.0}),
                        ("Horseshoe", {"tau_scale": 0.1, "lambda_scale": 1.0}),
                        ("Horseshoe", {"tau_scale": 0.1, "lambda_scale": 1.0, "delta_tau": True}),
                        ("Horseshoe", {"tau_scale": 1.0, "lambda_scale": 1.0, "regularized": True}),
                        ("SpikeNSlabLasso", {"lambda_spike": 20.0, "lambda_slab": 1.0, "relaxed_bernoulli": True, "temperature": 0.1}),
                    ]:
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
                            
                            # model.add_data(mdata)
                            # data_shuffled = model._data.values

                            # Load json file
                            with open(
                                str(filename).replace(".pkl", "_perf_timer.json"), "r"
                            ) as fp:
                                perf_timer = json.load(fp)

                        else:
                            print("Model not found")
                            continue
                        
                        if sparsity_prior is None:
                            sparsity_prior = "Normal"

                        # Measure R2
                        if hasattr(model._guide, "mode"):
                            z_hat = model._guide.mode("z")
                            w_hat = torch.cat(
                                [
                                    model._guide.mode(f"w_view_{m}").squeeze()
                                    for m in range(n_views)
                                ],
                                dim=1,
                            )
                        else:
                            z_hat = model._guide.median()["z"]
                            w_hat = torch.cat(
                                [
                                    model._guide.median()[f"w_view_{m}"].squeeze()
                                    for m in range(n_views)
                                ],
                                dim=1,
                            )

                        z_hat = z_hat.detach().cpu().numpy().squeeze()
                        w_hat = w_hat.detach().cpu().numpy().squeeze()
                        
                        # W activation
                        # Split w_hat into three equally sized parts along axis=1
                        w_hats = np.split(w_hat, 3, axis=1)
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
                                ][k] = compute_r2(
                                    mdata["feature_group_0"].X, x_hat_from_single_k
                                )[
                                    0
                                ] / mdata["feature_group_0"].X.size
                                
                                
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



#
# Plot #Active columns reconstruction with respect to features
#
BASE_DATA = df_w_act_l2

sns.set_theme(style="whitegrid")
sns.set_context("paper")
sns.set_palette(sns.color_palette("colorblind", 3))

plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["figure.autolayout"] = False
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.015

BASE_DATA_melt = BASE_DATA.melt(
    id_vars=["seed", "grid_features", "lr", "sparsity_prior", "view"],
    var_name="factor",
    value_name="L1 Norm",
)
# Replace factor names with numbers and convert to int
BASE_DATA_melt["factor"] = (
    BASE_DATA_melt["factor"].str.replace("factor_", "").astype(int)
)
# Plot L1 norms of W for each sparsity prior separately
fig, ax = plt.subplots(
    BASE_DATA_melt["sparsity_prior"].nunique(),
    1,
    figsize=(6.75, 15),
    sharex=False,
    sharey=False,
    tight_layout=True,
)
for idx, sparsity_prior in enumerate(BASE_DATA_melt["sparsity_prior"].unique()):
    # Draw a vertical line at N_FACTORS_TRUE
    ax[idx].axvline(x=N_SHARED_FACTORS + 0.5, color="gray", linestyle="--", linewidth=2)

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
        y="L1 Norm",
        hue="grid_features",
        ax=ax[idx],
        legend=False if idx < BASE_DATA_melt["sparsity_prior"].nunique() - 1 else True,
    )
    # Add title to each subplot with name of sparse prior
    ax[idx].set_title(f"Sparsity Prior: {sparsity_prior}")
    # Show only integer ticks on x axis from 1 to N_FACTORS_ESTIMATED
    ax[idx].set_xticks(range(1, N_FACTORS_ESTIMATED + 1))

# Set x and y labels
plt.xlabel("Factor")
plt.ylabel("L1 Norm of Factor")
# Place location below plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4)
plt.grid(True)
plt.show()
