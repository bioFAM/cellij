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

perf_r2_factors_all = Dict()
perf_r2_factors = Dict()
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

    for grid_features in tqdm([50, 100, 200, 400, 800, 1000, 2000, 5000, 10000]):
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
                ("SpikeNSlab", {"relaxed_bernoulli": True, "temperature": 0.1}),
                ("Nonnegativity", {}),
                ("SpikeNSlab", {"relaxed_bernoulli": False}),
                ("Horseshoe", {"tau_scale": 1.0, "lambda_scale": 1.0}),
                ("Lasso", {"lasso_scale": 0.1}),
                ("Horseshoe", {"tau_scale": 0.1, "lambda_scale": 1.0}),
                ("HorseshoeDeltaTau", {"tau_scale": 0.1, "lambda_scale": 1.0}),
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

                # Reconstruction
                # x_hat = np.matmul(z_hat, w_hat)
                # TODO: Compute reconstruction error

                # R2 of factors
                w_hats = np.split(w_hat, 3, axis=1)
                for idx, m in enumerate(mdata.mod):
                    w_true = mdata.mod[m].varm["w"]
                    avg, all_k, _, _ = compute_factor_correlation(w_true, w_hats[idx].T)
                    perf_r2_factors[seed][grid_features][lr][sparsity_prior] = avg
                    perf_r2_factors_all[seed][grid_features][lr][
                        sparsity_prior
                    ] = all_k  # TODO: Plot as well

                # Timer
                perf_time[seed][grid_features][lr][sparsity_prior] = perf_timer

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
                        ]


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
df = nested_dict_to_df(perf_r2_factors.to_dict()).reset_index()
for k, col_name in enumerate(["seed", "grid_features", "lr"]):
    # Rename kth columns
    df = df.rename(columns={f"level_{k}": col_name})

df = df.melt(
    id_vars=["seed", "grid_features", "lr"], var_name="sparsity_prior", value_name="r2"
)

df_time = nested_dict_to_df(perf_time.to_dict()).reset_index()
# Assign columns names from 1 to n
df_time.columns = [f"{x}" for x in range(1, len(df_time.columns) + 1)]
df_idx = df_time.iloc[:, :4]
# Coalesce all rows
df_idx = df_idx.assign(time=np.nansum(df_time.iloc[:, 4:].to_numpy(), axis=1))
df_idx.columns = ["seed", "grid_features", "lr", "sparsity_prior", "time"]


#
# Plot R2 of average factor reconstruction with respect to features
#
sns.set_theme(style="whitegrid")
g = sns.boxplot(
    data=df,
    x="grid_features",
    y="r2",
    hue="sparsity_prior",
).set(xlabel="Number of Features", ylabel="Avg. R2 of Factors")
plt.show()

#
# Plot Runtime with respect to features
#
sns.set_theme(style="whitegrid")
g = sns.boxplot(
    data=df_idx,
    x="grid_features",
    y="time",
    hue="sparsity_prior",
).set(xlabel="Number of Features", ylabel="Time [s]")
plt.show()

# # Split w_hat into three equally sized parts along axis=1
# w_hats = np.split(w_hat, 3, axis=1)
# w_activations = Dict()
# for m in range(3):
#     for k in range(w_hats[m].shape[0]):
#         w_activations[m][k] = np.sum(np.abs(w_hats[m][k]))

# df_activation = pd.DataFrame(w_activations)
# df_activation.columns = ["w_view_0", "w_view_1", "w_view_2"]
# for col in df_activation.columns:
#     print(col)
#     df_activation[col] = df_activation[col].sort_values(
#         ignore_index=True, ascending=False
#     )

# df_activation.plot()
