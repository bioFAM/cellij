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
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    device = torch.device("cpu")


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
N_SAMPLES = 200
MISSINGS = 0.0
# N_FACTORS_ESTIMATED = 20
OVERWRITE = False

# 20 Factors estimated for full grid  @ 200 samples
# 60 - 5 factors estimated for 400 and 1000 features @ 200 samples

PATH_DGP = "/home/m015k/code/cellij/experiments/sparsity_benchmark/data/"
PATH_MODELS = "/data/m015k/data/cellij/benchmark/benchmark_v1_missings/"

perf_r2_x = Dict()

for seed in [0, 1, 2, 3]:
    set_all_seeds(seed)

    for lr in [0.01, 0.001]:
        for grid_features in [500]:
            for N_FACTORS_ESTIMATED in [10]:
                n_samples = N_SAMPLES
                n_features = [grid_features, grid_features, grid_features]
                n_views = len(n_features)
                likelihoods = ["Normal" for _ in range(n_views)]

                n_fully_shared_factors = N_SHARED_FACTORS
                n_partially_shared_factors = N_PARTIAL_FACTORS
                n_private_factors = N_PRIVATE_FACTORS
                n_covariates = 0

                factor_size_params = (0.05, 0.15)
                s_factor_size_params = (
                    str(factor_size_params)
                    .replace(" ", "")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(",", "_")
                )

                if not (
                    Path(PATH_DGP)
                    .joinpath(
                        f"dgp_missing_{s_factor_size_params}_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                    )
                    .exists()
                ):
                    continue
                else:
                    print(f"Loading data from {PATH_DGP}...")
                    mdata = mudata.read(
                        Path(PATH_DGP).joinpath(
                            f"dgp_missing_{s_factor_size_params}_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                        )
                    )
                    mdata_no_missing = mdata.copy()

                for missing_percentage in [
                    0.02,
                    0.05,
                    0.1,
                    0.2,
                    0.35,
                    0.5,
                    # 0.65,
                    0.75,
                    0.9,
                    0.99,
                ]:
                    # Apply missings to data
                    if missing_percentage > 0:
                        for view, (name, anndata) in enumerate(mdata.mod.items()):
                            # Load missing file
                            random_indices = np.load(
                                Path(PATH_DGP).joinpath(
                                    f"random_idx_missing_{view}_{missing_percentage}_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.npy"
                                )
                            )

                            anndata_flattened = anndata.X.flatten()
                            anndata_flattened[random_indices] = np.nan
                            mdata[name].X = anndata_flattened.reshape(anndata.shape)

                    for sparsity_prior, prior_params in [
                        (None, {}),
                        # (
                        #     "SpikeNSlab",
                        #     {"relaxed_bernoulli": True, "temperature": 0.1},
                        # ),
                        # (
                        #     "SpikeNSlab",
                        #     {"relaxed_bernoulli": True, "temperature": 0.01},
                        # ),
                        ("SpikeNSlab", {"relaxed_bernoulli": False}),
                        ("Lasso", {"lasso_scale": 0.1}),
                        # ("Horseshoe", {"tau_scale": 1.0, "lambda_scale": 1.0}),
                        # ("Horseshoe", {"tau_scale": 0.1, "lambda_scale": 1.0}),
                        # ("HorseshoePlus", {"tau_const": 0.1, "eta_scale": 1.0}),
                        # (
                        #     "Horseshoe",
                        #     {
                        #         "tau_scale": 0.1,
                        #         "lambda_scale": 1.0,
                        #         "delta_tau": True,
                        #     },
                        # ),
                        (
                            "Horseshoe",
                            {
                                "tau_scale": 1.0,
                                "lambda_scale": 1.0,
                                "regularized": True,
                            },
                        ),
                        # (
                        #     "SpikeNSlabLasso",
                        #     {
                        #         "lambda_spike": 20.0,
                        #         "lambda_slab": 1.0,
                        #         "relaxed_bernoulli": True,
                        #         "temperature": 0.1,
                        #     },
                        # ),
                        # (
                        #     "SpikeNSlabLasso",
                        #     {
                        #         "lambda_spike": 20.0,
                        #         "lambda_slab": 0.01,
                        #         "relaxed_bernoulli": True,
                        #         "temperature": 0.1,
                        #     },
                        # ),
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
                            f"model_v1_features_missing_{missing_percentage}_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{sparsity_prior}_{N_FACTORS_ESTIMATED}_{lr}_{seed}_{s_params}.pkl"
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

                        d = (
                            model._guide.median()
                            if hasattr(model._guide, "median")
                            else model._guide.mode()
                        )
                        if any(["_view_" in x for x in d]):
                            query_key = "_view_"
                        else:
                            query_key = "_feature_group_"

                        if sparsity_prior in ["SpikeNSlabLasso"]:
                            z_hat = model._guide.median()["z"]
                            w_hat = torch.cat(
                                [
                                    (
                                        (
                                            1
                                            - model._guide.median()[
                                                f"lambda{query_key}{m}"
                                            ]
                                        )
                                        * model._guide.median()[
                                            f"lambda_spike{query_key}{m}"
                                        ]
                                        + model._guide.median()[f"lambda{query_key}{m}"]
                                        * model._guide.median()[
                                            f"lambda_slab{query_key}{m}"
                                        ]
                                    ).squeeze()
                                    for m in range(n_views)
                                ],
                                dim=1,
                            )

                        elif hasattr(model._guide, "mode"):
                            z_hat = model._guide.mode()["z"]
                            w_hat = torch.cat(
                                [
                                    model._guide.mode()[f"w{query_key}{m}"].squeeze()
                                    for m in range(n_views)
                                ],
                                dim=1,
                            )

                        else:
                            z_hat = model._guide.median()["z"]
                            w_hat = torch.cat(
                                [
                                    model._guide.median()[f"w{query_key}{m}"].squeeze()
                                    for m in range(n_views)
                                ],
                                dim=1,
                            )

                        temp_model = MOFA(n_factors=100, sparsity_prior="Horseshoe")
                        temp_model.add_data(data=mdata_no_missing, na_strategy=None)
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
                            "SpikeNSlab_relaxed_bernoulli=False": "SnS CB",
                            "SpikeNSlab_relaxed_bernoulli=True_temperature=0.1": "SnS RB (0.1)",
                            "SpikeNSlab_relaxed_bernoulli=True_temperature=0.01": "SnS RB (0.01)",
                            "SpikeNSlabLasso_lambda_spike=20.0_lambda_slab=1.0_relaxed_bernoulli=True_temperature=0.1": "SnS-L RB (20, 1.0, 0.1)",
                            "SpikeNSlabLasso_lambda_spike=20.0_lambda_slab=0.01_relaxed_bernoulli=True_temperature=0.1": "SnS-L RB (20, 0.01, 0.1)",
                        }
                        if sparsity_prior in legend_names:
                            sparsity_prior = legend_names[sparsity_prior]

                        # Reconstruction
                        x_hat = np.matmul(z_hat, w_hat)
                        x_hats = np.split(x_hat, 3, axis=1)
                        x_trues = np.split(x_true, 3, axis=1)

                        loss = 0.0
                        for view, (name, anndata) in enumerate(
                            mdata_no_missing.mod.items()
                        ):
                            # Load missing file
                            random_indices = np.load(
                                Path(PATH_DGP).joinpath(
                                    f"random_idx_missing_{view}_{missing_percentage}_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.npy"
                                )
                            )

                            true_elements = x_trues[view].flatten()[random_indices]
                            pred_elements = x_hats[view].flatten()[random_indices]
                            # Compute absolute error
                            loss += np.mean(np.abs(true_elements - pred_elements))
                            # loss += np.mean((true_elements - pred_elements) ** 2)

                        perf_r2_x[seed][grid_features][lr][sparsity_prior][
                            missing_percentage
                        ] = loss


# fig, ax = plt.subplots(1, 1, figsize=(20, 10))
# sns.heatmap(x_true, cmap="seismic", center=0)


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
df_r2 = nested_dict_to_df(perf_r2_x.to_dict()).reset_index()
for k, col_name in enumerate(["seed", "grid_features", "lr", "sparsity_prior"]):
    # Rename kth columns
    df_r2 = df_r2.rename(columns={f"level_{k}": col_name})

df_r2 = df_r2.melt(
    id_vars=["seed", "grid_features", "lr", "sparsity_prior"],
    var_name="missing_percentage",
    value_name="r2",
)
df_r2["missing_percentage"] = df_r2["missing_percentage"].astype(float)
# Drop all rows with missing values
df_r2 = df_r2.dropna()

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
# g = sns.scatterplot(
#     data=df_r2,
#     x="missing_percentage",
#     y="r2",
#     hue="sparsity_prior",
#     legend=False,
# ).set(xlabel="Percentage missing", ylabel="Loss of Reconstr. Missings")
g = sns.pointplot(
    data=df_r2,
    x="missing_percentage",
    y="r2",
    hue="sparsity_prior",
    dodge=0.4,
    join=False,
    # ci="sd"
    # legend=True,
    errorbar="sd",
).set(xlabel="Percentage missing", ylabel="Loss of Reconstr. Missings")
# Place location below plot
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.grid(True)
# Iterate over x-axis ticks and set rotation
v = [f"{int(100*float(x.get_text()))}%" for x in ax.get_xticklabels()]
ax.set_xticklabels(v, rotation=0, horizontalalignment="right")

# Define legend font size
plt.setp(ax.get_legend().get_texts(), fontsize="11")
# Save plot as pdf and png
plt.tight_layout()
# plt.savefig("plots/r2_features.pdf")
# plt.savefig("plots/r2_features.png")
plt.show()
