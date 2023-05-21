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
from functools import reduce
import anndata as ad
import muon as mu


import cellij
from cellij.core.models import MOFA
from cellij.utils import load_model, set_all_seeds

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    CUDA = torch.cuda.is_available()
    torch.cuda.set_device(1)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    device = torch.device("cpu")

PATH_DGP = "/data/m015k/data/cellij/benchmark/benchmark_cellij_missing_v2/"
PATH_MODELS = "/data/m015k/data/cellij/benchmark/benchmark_cellij_missing_v2/"

# Import CLL loader from Cellij
imp = cellij.core.Importer()
n_views = 4

perf_rec = Dict()

for seed in [0, 1, 2, 3, 4]:
    set_all_seeds(seed)

    for lr in [1e-3, 1e-4]:
        for n_factors_estimated in [10]:
            for percent_missing in [
                # 0.02,
                # 0.1,
                # 0.2,
                0.3,
                # 0.4,
                0.5,
                # # 0.6,
                0.7,
                # 0.8,
                # 0.9,
                # 0.95
                # 0.99,
            ]:
                # Loop over multiple sparsity priors
                for sparsity_prior, prior_params in [
                    (None, {}),
                    ("Lasso", {"lasso_scale": 0.1}),
                    (
                        "Horseshoe",
                        {
                            "tau_scale": 0.1,
                            "lambda_scale": 1.0,
                            "theta_scale": 1.0,
                            "delta_tau": False,
                            "regularized": False,
                            "ard": False,
                        },
                    ),
                    (
                        "Horseshoe",
                        {
                            "tau_scale": 0.1,
                            "lambda_scale": 1.0,
                            "theta_scale": 1.0,
                            "delta_tau": True,
                            "regularized": False,
                            "ard": False,
                        },
                    ),
                    (
                        "Horseshoe",
                        {
                            "tau_scale": 0.1,
                            "lambda_scale": 1.0,
                            "theta_scale": 1.0,
                            "delta_tau": False,
                            "regularized": True,
                            "ard": False,
                        },
                    ),
                    (
                        "Horseshoe",
                        {
                            "tau_scale": 0.1,
                            "lambda_scale": 1.0,
                            "theta_scale": 1.0,
                            "delta_tau": False,
                            "regularized": False,
                            "ard": True,
                        },
                    ),
                    (
                        "Horseshoe",
                        {
                            "tau_scale": 0.1,
                            "lambda_scale": 1.0,
                            "theta_scale": 1.0,
                            "delta_tau": False,
                            "regularized": True,
                            "ard": True,
                        },
                    ),
                    ("SpikeAndSlab", {"relaxed_bernoulli": True, "temperature": 0.1}),
                    ("SpikeAndSlab", {"relaxed_bernoulli": False}),
                    (
                        "SpikeAndSlabLasso",
                        {
                            "lambda_spike": 20.0,
                            "lambda_slab": 0.01,
                            "relaxed_bernoulli": True,
                            "temperature": 0.1,
                        },
                    ),
                ]:
                    print(
                        f" - {sparsity_prior} | {prior_params} | {lr} | {n_factors_estimated} | {seed} | {percent_missing}"
                    )
                    # Combine all parameters used for the prior into a string
                    # This allows to train model with the same prior but different parameters
                    s_params = (
                        "_".join([f"{k}={v}" for k, v in prior_params.items()])
                        if prior_params
                        else "None"
                    )
                    filename = Path(PATH_MODELS).joinpath(
                        f"model_cll_{seed}_{percent_missing}_{sparsity_prior}_{n_factors_estimated}_{lr}_{s_params}.pkl"
                    )

                    if Path(filename).exists():
                        print(f"Loading {filename}")
                        model = load_model(str(filename))
                    else:
                        print("Model not found")
                        continue

                    query_key = "_feature_group_"

                    if sparsity_prior in ["SpikeAndSlabLasso"]:
                        print(model._guide.median().keys())
                        z_hat = model._guide.median()["z"]
                        w_hat = torch.cat(
                            [
                                (
                                    (1 - model._guide.median()[f"lambda{query_key}{m}"])
                                    * model._guide.median()[f"w_spike{query_key}{m}"]
                                    + model._guide.median()[f"lambda{query_key}{m}"]
                                    * model._guide.median()[f"w_slab{query_key}{m}"]
                                ).squeeze()
                                for m in range(n_views)
                            ],
                            dim=1,
                        )

                    elif hasattr(model._guide, "mode"):
                        z_hat = model._guide.mode()["z"]
                        w_hat = torch.cat(
                            [
                                model._guide.mode()[m].squeeze()
                                for m in [
                                    "w_drugs",
                                    "w_methylation",
                                    "w_mrna",
                                    "w_mutations",
                                ]
                            ],
                            dim=1,
                        )

                    else:
                        z_hat = model._guide.median()["z"]

                        w_hat = torch.cat(
                            [
                                model._guide.median()[m].squeeze()
                                for m in [
                                    "w_drugs",
                                    "w_methylation",
                                    "w_mrna",
                                    "w_mutations",
                                ]
                            ],
                            dim=1,
                        )

                    z_hat = z_hat.detach().cpu().numpy().squeeze()
                    w_hat = w_hat.detach().cpu().numpy().squeeze()

                    if sparsity_prior is None:
                        sparsity_prior = "Normal"
                    else:
                        sparsity_prior = sparsity_prior + "_" + s_params

                    legend_names = {
                        "Horseshoe_tau_scale=0.1_lambda_scale=1.0_theta_scale=1.0_delta_tau=False_regularized=False_ard=False": "HS(0.1,1,1)",
                        "Horseshoe_tau_scale=0.1_lambda_scale=1.0_theta_scale=1.0_delta_tau=False_regularized=False_ard=True": "HS(0.1,1,1)+ARD",
                        "Horseshoe_tau_scale=0.1_lambda_scale=1.0_theta_scale=1.0_delta_tau=False_regularized=True_ard=False": "HS Reg(0.1,1,1)",
                        "Horseshoe_tau_scale=0.1_lambda_scale=1.0_theta_scale=1.0_delta_tau=False_regularized=True_ard=True": "HS Reg(0.1,1,1)+ARD",
                        "Horseshoe_tau_scale=0.1_lambda_scale=1.0_theta_scale=1.0_delta_tau=True_regularized=False_ard=False": "HS ConstTau(0.1,1,1)",
                        "Lasso_lasso_scale=0.1": "Laplace(0,0.1)",
                        "Normal": "Normal",
                        "SpikeAndSlabLasso_lambda_spike=20.0_lambda_slab=0.01_relaxed_bernoulli=True_temperature=0.1": "SnSLasso RB(20,0.01,0.1)",
                        "SpikeAndSlab_relaxed_bernoulli=False": "SnS CB",
                        "SpikeAndSlab_relaxed_bernoulli=True_temperature=0.1": "SnS RB(0.1)",
                    }
                    if sparsity_prior in legend_names:
                        sparsity_prior = legend_names[sparsity_prior]

                    # Reconstruction
                    x_hat = np.matmul(z_hat, w_hat)

                    # Load a fresh dataset
                    imp = cellij.core.Importer()
                    mdata = imp.load_CLL()
                    # Use the intersection of observations
                    obs_names = mdata.obs.index.values
                    common_obs = reduce(
                        np.intersect1d,
                        [v.obs_names.values for k, v in mdata.mod.items()],
                    )
                    mods = {
                        name: data[common_obs].copy()
                        for name, data in mdata.mod.items()
                    }
                    mdata_new = mu.MuData(mods)

                    nan_idx = set()
                    for modality, data in mdata_new.mod.items():
                        print(modality)
                        nan_idx |= set(data[np.isnan(data.X).all(axis=1)].obs.index)
                    nan_idx = list(nan_idx)
                    non_nan_idx = list(set(mdata_new.obs.index) - set(nan_idx))

                    mods = {
                        name: data[non_nan_idx].copy()
                        for name, data in mdata_new.mod.items()
                    }
                    mdata = mu.MuData(mods)

                    mdata_no_nan = mdata.copy()

                    # # Use the intersection of observations
                    # obs_names = mdata.obs.index.values
                    # common_obs = reduce(
                    #     np.intersect1d,
                    #     [v.obs_names.values for k, v in mdata.mod.items()],
                    # )
                    # mods = {
                    #     name: data[common_obs].copy()
                    #     for name, data in mdata.mod.items()
                    # }

                    # mdata_new = mu.MuData(mods)

                    # nan_idx = set()
                    # for modality, data in mdata_new.mod.items():
                    #     print(modality)
                    #     nan_idx |= set(data[np.isnan(data.X).all(axis=1)].obs.index)
                    # nan_idx = list(nan_idx)
                    # non_nan_idx = list(set(mdata_new.obs.index) - set(nan_idx))

                    # mods = {
                    #     name: data[non_nan_idx].copy()
                    #     for name, data in mdata_new.mod.items()
                    # }
                    # mdata_new = mu.MuData(mods)

                    # Loop over all modalities and replace x percent of the values with missings
                    for idx, (modality, anndata) in enumerate(mdata.mod.items()):
                        # print(modality)
                        # Load random indices if exists
                        filename_data = Path(PATH_DGP).joinpath(
                            f"model_cll_random_indices_{seed}_{percent_missing}_{modality}.npy"
                        )
                        if filename_data.exists():
                            # print(f"Loading random indices from {filename_data}")
                            random_indices = np.load(filename_data)
                        else:
                            pass
                            n_features, n_samples = anndata.shape[0], anndata.shape[1]
                            num_elements = n_features * n_samples

                            already_missing_indices = np.where(
                                np.isnan(anndata.X.flatten())
                            )[0]
                            values_potential_missing_indices = list(
                                set(np.arange(num_elements))
                                - set(already_missing_indices)
                            )

                            num_values_to_set_missing = int(
                                percent_missing * len(values_potential_missing_indices)
                            )

                            random_indices = np.random.choice(
                                values_potential_missing_indices,
                                size=num_values_to_set_missing,
                                replace=False,
                            )

                            # Save random_indices to file
                            print(f"Saving random indices to {filename_data}")
                            np.save(filename_data, random_indices)

                        # print("Missings before: ", np.sum(np.isnan(anndata.X)))
                        anndata_flattened = anndata.X.flatten()
                        anndata_flattened[random_indices] = np.nan
                        mdata[modality].X = anndata_flattened.reshape(anndata.shape)
                        # print("Missings after: ", np.sum(np.isnan(anndata.X)))

                        # Center features at zero mean
                        mdata[modality].X = mdata[modality].X - np.nanmean(
                            mdata[modality].X, axis=0
                        )

                        # Center features at zero mean
                        mdata_no_nan[modality].X = mdata_no_nan[
                            modality
                        ].X - np.nanmean(mdata_no_nan[modality].X, axis=0)

                    temp_model = MOFA(n_factors=100, sparsity_prior="Horseshoe")
                    temp_model.add_data(data=mdata, na_strategy=None)
                    x_true = temp_model._data.values

                    temp_model_2 = MOFA(n_factors=100, sparsity_prior="Normal")
                    temp_model_2.add_data(data=mdata_no_nan, na_strategy=None)
                    x_true_no_add_nan = temp_model_2._data.values

                    # Check where nan values are in x_true but not in x_true_no_add_nan
                    # but only in the "drugs" view
                    nan_diff = np.isnan(x_true[:, :310]) & ~np.isnan(
                        x_true_no_add_nan[:, :310]
                    )
                    loss = np.sqrt(
                        np.mean(
                            (
                                x_hat[:, :310][nan_diff]
                                - x_true_no_add_nan[:, :310][nan_diff]
                            )
                            ** 2
                        )
                    )

                    # nan_diff = np.isnan(x_true) & ~np.isnan(x_true_no_add_nan)
                    # loss = np.sqrt(
                    #     np.mean((x_hat[nan_diff] - x_true_no_add_nan[nan_diff]) ** 2)
                    # )

                    perf_rec[percent_missing][sparsity_prior][n_factors_estimated][lr][
                        seed
                    ] = loss


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


df_rec = nested_dict_to_df(perf_rec.to_dict()).reset_index()
# Assign columns names from 1 to n
df_rec.columns = [f"{x}" for x in range(1, len(df_rec.columns) + 1)]
df_rec.columns = ["missings", "sparsity_prior", "factors", "lr"] + [
    f"seed_{x}" for x in range(df_rec.shape[1] - 4)
]
df_rec = df_rec.melt(
    id_vars=["missings", "sparsity_prior", "factors", "lr"],
    # value_vars=[f"seed_{x}" for x in range(df_rec.shape[1] - 5)],
    # var_name="seed",
    value_name="loss",
)


plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

fig, ax = plt.subplots(1, 1, figsize=(6.75, 5))
# Do not plot outliers
g = sns.lineplot(
    data=df_rec,
    x="missings",
    y="loss",
    hue="sparsity_prior",
).set(xlabel="Percentage Missing", ylabel="Reconstruction Loss")
# Place location below plot
plt.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=4)
plt.grid(True)
# Define legend font size
plt.setp(ax.get_legend().get_texts(), fontsize="11")
# Show xticks labels as percent
vals = ax.get_xticks()
ax.set_xticklabels(["{:,.0%}".format(x) for x in vals])
# Set y range to 0 to 1
# plt.ylim(0.2, 1.05)
# Save plot as pdf and png
plt.tight_layout()
# plt.savefig("plots/r2_features.pdf")
# plt.savefig("plots/r2_features.png")
plt.show()

df_rec.groupby(["missings", "sparsity_prior", "factors", "lr"])["loss"].mean()