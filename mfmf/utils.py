from __future__ import annotations

import os
import mfmf
import pyro
import torch
import anndata
import logging
import datetime
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import Optional, List, Union
from scipy.optimize import linear_sum_assignment

from mfmf.modules.data import MergedView


def get_current_gpu_usage(output: str = "df") -> str:

    device_name = torch.cuda.get_device_name(device="cuda:0")

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv"],
        stdout=subprocess.PIPE,
    )

    result = result.stdout.decode("utf-8").split(os.linesep)
    del result[-1]  # remove last empty line

    result_df = pd.DataFrame(columns=result[0].split(","))

    for gpu in result[1:]:
        tmp = {}
        for idx, val in enumerate(result[0].split(",")):
            tmp[val] = [gpu.split(",")[idx]]
        result_df = pd.concat([result_df, pd.DataFrame(tmp)])

    result_df = result_df[result_df.name == device_name]

    if output == "df":

        return result_df

    elif output == "formatted":

        output_str = result_df.head(1).values.tolist()[0]
        output_str = [x.strip() for x in output_str]
        output_str = (
            output_str[0]
            + " ("
            + output_str[1].replace(" MiB", "")
            + "/"
            + output_str[2]
            + ")"
        )

        return output_str


def minmax_scale(
    data: List[np.ndarray, torch.Tensor]
) -> List[np.ndarray, torch.Tensor]:
    """Scales a matrix to the range [0, 1], returns same dtype as input

    Needed since the stupid sklearn preprocessing module results in a case where
    a float(1.000) is greater than 1 due to float precision issues.

    """

    if not isinstance(data, (torch.Tensor, np.ndarray)):
        raise TypeError("Parameter 'data' must be a numpy.ndarray or torch.Tensor.")

    if isinstance(data, torch.Tensor):

        data = data.cpu()
        data = data - torch.min(data, dim=1, keepdim=True)[0]
        data = data / torch.max(data, dim=1, keepdim=True)[0]

        return data

    elif isinstance(data, np.ndarray):

        data = data - np.min(data, axis=1, keepdims=True)
        data = data / np.max(data, axis=1, keepdims=True)

        return data


def merge_views(
    views: List[mfmf.modules.data.View],
) -> Optional[torch.Tensor]:
    """Merges multiple views into a single tensor."""

    if isinstance(views, list):
        if not all([isinstance(view, mfmf.modules.data.View) for view in views]):
            raise TypeError("All elements of 'views' must be of type 'mfmf.data.View'.")
    else:
        raise TypeError("Parameter 'views' must be a 'list' of 'mfmf.data.View' views.")

    view_dfs = []
    feature_names = []
    n_features = []
    view_names = []

    for view in views:

        view_dfs.append(
            pd.DataFrame(
                data=view.data.cpu().numpy(),
                index=view.obs_names,
                columns=view.feature_names,
            )
        )
        feature_names += view.feature_names
        n_features.append(len(view.feature_names))
        view_names += [view.name]

    if not (len(set(feature_names)) == len(feature_names)):
        raise ValueError("Duplicate feature_names found in the passed views.")

    df_out = view_dfs[0]

    for i in range(1, len(view_dfs)):

        df_out = df_out.merge(
            right=view_dfs[i],
            left_index=True,
            right_index=True,
            how="outer",
        )

    df_out = anndata.AnnData(df_out)

    offset_borders = [0] + np.cumsum(n_features).tolist()
    feature_offsets = {}
    for idx, v in enumerate(view_names):

        feature_offsets[v] = (offset_borders[idx], offset_borders[idx + 1])

    merge_view = MergedView(
        data=df_out,
        feature_offsets=feature_offsets,
    )

    return merge_view


def plot_optimal_assignment(
    x1: torch.tensor,
    x2: torch.tensor,
    assignment_dim: int,
    nrows: int = 1,
    ncols: int = None,
    figsize: list = [8, 4],
    cmap: str = "seismic",
    center: int = 0,
):
    """Generates a pairwise plot of the 'optimally matched' columns of two 2D tensors.

    Args:

        x1 (torch.tensor): first (fixed) tensor.
        x2 (torch.tensor): second tensor.
        assignment_dim (int): assignment dimension.
        nrows (int, optional): Passed to plt.subplots(). Defaults to 1.
        ncols (int, optional): Passed to plt.subplots(). Defaults to None.
        figsize (tuple, optional):Passed to plt.subplots(). Defaults to (8, 4).
        cmap (str, optional): Passed to sns.heatmap(). Defaults to "seismic".
        center (int, optional): Passed to sns.heatmap(). Defaults to 0.

    Example:

        $ t1 = torch.tensor(np.random.normal(size=[5, 20]))
        $ t2 = torch.tensor(np.random.normal(size=[5, 20]))
        $ plot_optimal_assignment(t1, t2, 0)
    """

    if not torch.is_tensor(x1):
        if isinstance(x1, pd.DataFrame):
            x1 = x1.values
        x1 = torch.tensor(x1, device="cpu")

    if not torch.is_tensor(x2):
        if isinstance(x2, pd.DataFrame):
            x2 = x2.values
        x2 = torch.tensor(x2, device="cpu")

    if (len(x1.shape) != 2) or (len(x2.shape) != 2):
        raise ValueError("Only 2D tensors are supported.")

    if x1.shape == x2.T.shape:
        x2 = x2.T
        logging.warning("Had to transpose 'x2' to match 'x1'.")

    if not x1.shape == x2.shape:
        raise ValueError("The dimensions of 'x1' and 'x2' must match.")

    ncols = x1.shape[assignment_dim]
    figsize[0] = 3 + 3 * ncols

    if not isinstance(nrows, int):
        raise TypeError("Paramter 'nrows' must be a positive integer.")

    if not nrows >= 1:
        raise ValueError("Paramter 'nrows' must be a positive integer.")

    if not isinstance(ncols, int):
        raise TypeError("Paramter 'ncols' must be a positive integer.")

    if not ncols >= 1:
        raise ValueError("Paramter 'ncols' must be a positive integer.")

    ind, cor = get_optimal_assignment(x1=x1, x2=x2, assignment_dim=assignment_dim)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for idx, i in enumerate(ind):

        if assignment_dim == 1:
            matched_array = np.stack((x1[:, idx], x2[:, i]), axis=-1)
        elif assignment_dim == 0:
            matched_array = np.stack((x1[idx, :], x2[i, :]), axis=-1)

        sns.heatmap(
            matched_array,
            cmap=cmap,
            center=center,
            ax=axs[idx],
            vmin=min(x1.min(), x2.min()),
            vmax=max(x1.max(), x2.max()),
        )
        axs[idx].set_title(f"corr = {str(np.round(cor[idx], 4))}")

    fig.suptitle(
        f"total correlation = {str(np.round(np.mean(np.abs(cor)), 4))}",
        size=18,
        x=0.1,
        y=1.05,
        ha="left",
    )


def get_optimal_assignment(
    x1: torch.tensor,
    x2: torch.tensor,
    assignment_dim: int,
):
    """Find the permutation of x2 elements along the selected dimension to
        maximize correlation with x1.

    Args:
        x1: first (fixed) tensor
        x2: second tensor
        assignment_dim: assignment dimension

    Returns:
        optimal assignment indices of x2 items, correlation coefficients
    """

    if not torch.is_tensor(x1):
        if isinstance(x1, pd.DataFrame):
            x1 = x1.values
        x1 = torch.tensor(x1, device="cpu")
    if not torch.is_tensor(x2):
        if isinstance(x2, pd.DataFrame):
            x2 = x2.values
        x2 = torch.tensor(x2, device="cpu")

    correlation = torch.zeros(
        [x1.shape[assignment_dim], x2.shape[assignment_dim]], device="cpu"
    )
    for i in range(x1.shape[assignment_dim]):
        for j in range(x2.shape[assignment_dim]):
            correlation[i, j] = pearsonr(
                torch.narrow(x1, assignment_dim, i, 1).flatten(),
                torch.narrow(x2, assignment_dim, j, 1).flatten(),
            )[0]
    correlation = torch.nan_to_num(correlation, 0)

    row_ind, col_ind = linear_sum_assignment(-1 * torch.abs(correlation))
    return col_ind, correlation[row_ind, col_ind].numpy()


def get_formatted_number(
    number: float, n_before_separator: int, n_after_separator: int = 3
) -> str:

    # if n_before_separator < len(str(int(number))):
    #     raise ValueError(
    #         "Parameter 'n_before_separator' must be larger or equal to the floor of 'number'."
    #     )

    formatted_number = (
        "{:"
        + str(n_before_separator + n_after_separator)
        + "."
        + str(n_after_separator)
        + "f}"
    ).format(number)

    return formatted_number


def get_all_param_values(model: mfmf.modules.core.FactorModel, param_name: str):

    if param_name not in model.pyro_params.keys():
        raise ValueError(
            "Parameter "
            + param_name
            + " is not stored in the given model. Available choices are: \n- "
            + "\n- ".join([k for k in model.pyro_params.keys()])
        )

    param_dict = {}
    for k, v in model.pyro_params[param_name].items():
        param_dict[k] = v

    return param_dict


def replace_nan(
    data: Union[np.ndarray, torch.Tensor],
    nan_replacement: float = 0.0,
    view_name: str = None,
) -> torch.Tensor:

    if not isinstance(data, (np.ndarray, torch.Tensor)):
        raise TypeError("Parameter 'data' must be a 'numpy.ndarray' or 'torch.tensor'.")

    if not isinstance(nan_replacement, float):
        if isinstance(nan_replacement, int):
            nan_replacement = float(nan_replacement)
        else:
            raise TypeError("Parameter 'nan_replacement' must be a float.")

    # Construct warning message in case we need it.
    w = "Found NaN"
    if view_name is not None:
        if (
            view_name == "mfmf_merging_mfmf"
        ):  # weird string that the user shouldn't accidentally use
            w += " in merged data"
        else:
            w += f" in input data for view {view_name}"

    w += ", replacing NaN with " + str(nan_replacement) + "."

    if isinstance(data, np.ndarray):
        if np.isnan(data).any():
            data = np.nan_to_num(x=data, nan=nan_replacement)
        else:
            w = None
    elif isinstance(data, torch.Tensor):
        if torch.isnan(data).any():
            data = torch.nan_to_num(input=data, nan=nan_replacement)
        else:
            w = None

    if w is not None:
        logging.warning(w)

    return data


def get_matrix_with_prior(
    id: str, prior: str, feature_plate=None, obs_plate=None, factor_plate=None, **kwargs
) -> dict:
    """Returns pyro.sample object with the associated priors as a dict.

    Initiates a Pyro sample statement for a matrix with the appropriate
    sparsity pattern and returns all constituents as a dict.

    Args:
        id: identifier to use for associated objects, f.e. 'mrna'
        type: which sparsity prior to use, f.e. 'horseshoe'
        shape: shape of the Pyro sample statement

    Example:
    >>> get_matrix_with_prior(
            name="w",
            id="mrna_all_genes",
            type="horseshoe"
            shape=[self.feature_plate.size, self.factor_plate.size]
        )
        ...
        {
            "w_mrna_all_genes": pyro.sample(name="w_mrna_all_genes", ...),
            "hs_lambda_mrna_all_genes": pyro.sample(name="hs_lambda_mrna_all_genes", ...),
            "hs_tau_mrna_all_genes": pyro.sample(name="hs_tau_mrna_all_genes", ...),
        }
    """

    if not isinstance(id, str):
        raise TypeError("Parameter 'id' must be a string.")

    valid_prior = [
        "none",
        "horseshoe",
        "finnish_horseshoe",
        "ard",
        "spike-and-slab",
        "ard_spike-and-slab",
    ]

    if not isinstance(prior, str) or prior not in valid_prior:
        raise ValueError(
            "Parameter 'prior' must be one of '" + "', '".join(valid_prior) + "'."
        )

    if feature_plate is not None and not isinstance(feature_plate, pyro.plate):
        raise TypeError(
            "If present, parameter 'feature_plate' must be a pyro.plate object."
        )

    if obs_plate is not None and not isinstance(obs_plate, pyro.plate):
        raise TypeError(
            "If present, parameter 'obs_plate' must be a pyro.plate object."
        )

    if factor_plate is not None and not isinstance(factor_plate, pyro.plate):
        raise TypeError(
            "If present, parameter 'factor_plate' must be a pyro.plate object."
        )

    if feature_plate is None and obs_plate is None:
        raise ValueError(
            "At least one of 'feature_plate' and 'obs_plate' must be given."
        )

    # TODO(ttreis): Is there a scenario in which I only give obs_plate and feature_plate?

    res = {}
    hs_scaler = 0.9

    # Reference the correct plates in plate_a/_b so that I can use the same
    # line of code for double-broadcasting
    if feature_plate is not None:

        name = "w"
        plate_a = feature_plate
        plate_b = factor_plate

        if "feature_reg_kwargs" in kwargs.keys():
            if "hs_scaler" in kwargs["feature_reg_kwargs"].keys():
                hs_scaler = kwargs["feature_reg_kwargs"]["hs_scaler"]

    elif obs_plate is not None:

        name = "z"
        plate_a = obs_plate
        plate_b = factor_plate

        if "sample_reg_kwargs" in kwargs.keys():
            if "hs_scaler" in kwargs["sample_reg_kwargs"].keys():
                hs_scaler = kwargs["sample_reg_kwargs"]["hs_scaler"]

    if prior == "horseshoe":

        res["hs_tau"] = pyro.sample(
            name=f"hs_tau_{id}",
            fn=pyro.distributions.HalfCauchy(
                scale=torch.Tensor([0.1])  # arbitrary number
            ),
        )

        with plate_a, plate_b:

            res["hs_lambda"] = pyro.sample(
                name=f"hs_lambda_{id}",
                fn=pyro.distributions.HalfCauchy(
                    scale=torch.Tensor([0.1])  # arbitrary number
                ),
            )

            res["lambda"] = res["hs_tau"] * res["hs_lambda"] * hs_scaler

            res[name] = pyro.sample(
                name=f"{name}_{id}",
                fn=pyro.distributions.Normal(
                    loc=torch.zeros(1),
                    scale=res["lambda"],
                ),
            )

    elif prior == "finnish_horseshoe":

        res["fhs_tau"] = pyro.sample(
            name=f"fhs_tau_{id}",
            fn=pyro.distributions.HalfCauchy(
                scale=torch.Tensor([0.1])  # arbitrary number
            ),
        )

        res["fhs_c_sq"] = pyro.sample(
            name=f"fhs_c_sq_{id}",
            fn=pyro.distributions.InverseGamma(
                concentration=torch.tensor([0.5]), rate=torch.tensor([0.5])
            ),
        )

        with plate_a, plate_b:

            res["fhs_lambda"] = pyro.sample(
                name=f"fhs_lambda_{id}",
                fn=pyro.distributions.HalfCauchy(
                    scale=torch.Tensor([0.1])  # arbitrary number
                ),
            )

            res[f"{name}_scale"] = (
                res["fhs_tau"]
                * torch.sqrt(
                    (res["fhs_c_sq"] * torch.pow(res["fhs_lambda"], 2))
                    / (
                        res["fhs_c_sq"]
                        + torch.pow(
                            res["fhs_tau"] * res["fhs_lambda"],
                            2,
                        )
                    )
                )
                * hs_scaler
            )

            res[name] = pyro.sample(
                name=f"{name}_{id}",
                fn=pyro.distributions.Normal(
                    loc=torch.zeros(1),
                    scale=res[f"{name}_scale"],
                ),
            )

    elif prior == "ard":

        with factor_plate:

            res["ss_theta"] = torch.ones(1)

            res["ss_alpha"] = pyro.sample(
                name=f"ss_alpha_{id}",
                fn=pyro.distributions.InverseGamma(
                    concentration=torch.Tensor([0.00001]),
                    rate=torch.Tensor([0.00001]),
                ),
            )

        with plate_a, plate_b:

            res[f"{name}_norm"] = pyro.sample(
                name=f"{name}_norm_{id}",
                fn=pyro.distributions.Normal(
                    loc=torch.zeros(1),
                    scale=res["ss_alpha"],
                ),
            )

            res[f"{name}_ber"] = pyro.sample(
                name=f"{name}_ber_{id}",
                fn=pyro.distributions.ContinuousBernoulli(probs=res["ss_theta"]),
            )

            res[name] = res[f"{name}_norm"] * res[f"{name}_ber"]

    elif prior == "spike-and-slab":

        with factor_plate:

            res["ss_theta"] = pyro.sample(
                name=f"ss_theta_{id}",
                fn=pyro.distributions.Beta(
                    concentration0=torch.Tensor([0.5]),
                    concentration1=torch.Tensor([0.5]),
                ),
            )

            res["ss_alpha"] = torch.ones(1)

        with plate_a, plate_b:

            res[f"{name}_norm"] = pyro.sample(
                name=f"{name}_norm_{id}",
                fn=pyro.distributions.Normal(
                    loc=torch.zeros(1),
                    scale=res["ss_alpha"],
                ),
            )

            res[f"{name}_ber"] = pyro.sample(
                name=f"{name}_ber_{id}",
                fn=pyro.distributions.ContinuousBernoulli(probs=res["ss_theta"]),
            )

            res[name] = res[f"{name}_norm"] * res[f"{name}_ber"]

    elif prior == "ard_spike-and-slab":

        with factor_plate:

            res["ss_theta"] = pyro.sample(
                name=f"ss_theta_{id}",
                fn=pyro.distributions.Beta(
                    concentration0=torch.Tensor([0.5]),
                    concentration1=torch.Tensor([0.5]),
                ),
            )

            res["ss_alpha"] = pyro.sample(
                name=f"ss_alpha_{id}",
                fn=pyro.distributions.InverseGamma(
                    concentration=torch.Tensor([0.00001]),
                    rate=torch.Tensor([0.00001]),
                ),
            )

        with plate_a, plate_b:

            res[f"{name}_norm"] = pyro.sample(
                name=f"{name}_norm_{id}",
                fn=pyro.distributions.Normal(
                    loc=torch.zeros(1),
                    scale=res["ss_alpha"],
                ),
            )

            res[f"{name}_ber"] = pyro.sample(
                name=f"{name}_ber_{id}",
                fn=pyro.distributions.ContinuousBernoulli(probs=res["ss_theta"]),
            )

            res[name] = res[f"{name}_norm"] * res[f"{name}_ber"]

    elif prior == "none":

        with plate_a, plate_b:

            res[name] = pyro.sample(
                name=f"{name}_{id}",
                fn=pyro.distributions.Normal(loc=torch.zeros(1), scale=torch.ones(1)),
            )

    else:

        raise NotImplementedError("Not implemented.")

    return res


# def generate_benchmark_data(
#     dist=pyro.distributions.Normal(0, 1),
#     rows: int = 100,
#     columns: int = 100,
#     factors: int = 10,
#     missing_perc: int = 0,
#     seed: int = None,
#     base_name: str = None,
#     output_dir: str = None,
# ):

#     if not isinstance(rows, int):
#         raise TypeError("Parameter 'rows' must be of type 'int'.")

#     if not isinstance(columns, int):
#         raise TypeError("Parameter 'columns' must be of type 'int'.")

#     if not isinstance(factors, int):
#         raise TypeError("Parameter 'factors' must be of type 'int'.")

#     if not isinstance(missing_perc, int):
#         raise TypeError("Parameter 'missing_perc' must be of type 'int'.")

#     if not isinstance(seed, int):
#         raise TypeError("Parameter 'seed' must be of type 'int'.")

#     if not isinstance(output_dir, str):
#         raise TypeError("Parameter 'output_dir' must be of type 'str'.")

#     # lock in random seeds
#     if isinstance(seed, int):
#         rnd_seed = seed
#     else:
#         ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
#         rnd_seed = sum([int(x) for x in ts])  # sum of all digits in timestamp

#     np.random.seed(rnd_seed)
#     torch.manual_seed(rnd_seed)

#     if base_name is None:

#         filename = output_dir + "data"  # + ts
#         filename += f"_{dist.__name__}likelihood"
#         filename += f"_{rnd_seed}seed"
#         filename += f"_{rows}rows"
#         filename += f"_{columns}columns"
#         filename += f"_{factors}factors"

#     else:

#         filename = output_dir + base_name

#     # w = dist.sample(sample_shape=[rows, factors])
#     # z = dist.sample(sample_shape=[factors, columns])
#     w = np.random.normal(0, 1, size=(rows, factors))
#     z = np.random.normal(0, 1, size=(factors, columns))

#     if missing_perc != 0:

#         n_missing = int((rows * columns) * (missing_perc / 100))

#         mask = np.zeros(rows * factors, dtype=bool)
#         mask[:n_missing] = True
#         np.random.shuffle(mask)
#         mask = mask.reshape(rows, factors)

#         w[mask] = np.nan

#         mask = np.zeros(factors * columns, dtype=bool)
#         mask[:n_missing] = True
#         np.random.shuffle(mask)
#         mask = mask.reshape(factors, columns)

#         z[mask] = np.nan

#     w = torch.tensor(w)
#     z = torch.tensor(z)

#     eps = dist.sample(sample_shape=[rows, columns])
#     y = torch.matmul(w, z) + eps

#     # save the data
#     np.savetxt(filename + "_w.csv", w, delimiter=";", fmt="%1.5f")
#     np.savetxt(filename + "_z.csv", z, delimiter=";", fmt="%1.5f")
#     np.savetxt(filename + "_eps.csv", eps, delimiter=";", fmt="%1.5f")
#     np.savetxt(filename + "_y.csv", y, delimiter=";", fmt="%1.5f")


def generate_benchmark_data(
    w_dist=pyro.distributions.Normal(0, 1),
    z_dist=pyro.distributions.Normal(0, 1),
    rows: int = 100,
    columns: int = 100,
    factors: int = 10,
    factors_active: list[bool] = None,
    sparsity: dict = {"type": "signal_per_factor", "n_signals": (20, 40)},
    seed: int = None,
    base_name: str = None,
    output_dir: str = "./",
    device: str = "cpu",
):

    if not isinstance(rows, int):
        raise TypeError("Parameter 'rows' must be of type 'int'.")

    if not isinstance(columns, int):
        raise TypeError("Parameter 'columns' must be of type 'int'.")

    if not isinstance(factors, int):
        raise TypeError("Parameter 'factors' must be of type 'int'.")

    if not isinstance(seed, int):
        raise TypeError("Parameter 'seed' must be of type 'int'.")

    if not isinstance(output_dir, str):
        raise TypeError("Parameter 'output_dir' must be of type 'str'.")

    if not isinstance(device, str):
        raise TypeError("Parameter 'device' must be of type 'str'.")

    if factors_active is not None:
        assert len(factors_active) == factors
        assert all([isinstance(x, bool) for x in factors_active])

    # lock in random seeds
    if isinstance(seed, int):
        rnd_seed = seed
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d%H%M")
        rnd_seed = sum([int(x) for x in ts])  # sum of all digits in timestamp

    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    if base_name is None:

        filename = output_dir + "data"  # + ts
        filename += f"_{w_dist.__str__().split('(')[0]}w"
        filename += f"_{z_dist.__str__().split('(')[0]}z"
        filename += f"_{rnd_seed}seed"
        filename += f"_{rows}rows"
        filename += f"_{columns}columns"
        filename += f"_{factors}factors"

    else:

        filename = output_dir + base_name

    # make epsilon
    eps = []
    for _ in range(columns):
        tau_for_col = torch.sqrt(pyro.distributions.Gamma(50, 1).sample())
        eps_for_col = pyro.distributions.Normal(0, 1 / tau_for_col).sample(
            sample_shape=[rows]
        )
        eps.append(eps_for_col)
    eps = torch.cat(eps, dim=0).reshape(rows, columns)

    # make w
    w = []
    for i in range(factors):

        if factors_active is not None and not factors_active[i]:

            w.append(torch.zeros(columns))

        else:

            n_signal = np.random.randint(
                low=sparsity["n_signals"][0], high=sparsity["n_signals"][1], size=1
            )
            factor_in_w_signal = [x for x in w_dist.sample(sample_shape=[n_signal[0]])]

            lower_bound = -0.5
            upper_bound = 0.5
            # remove anything within [lower_bound, upper_bound]
            while any([lower_bound < x < upper_bound for x in factor_in_w_signal]):
                for x in factor_in_w_signal:
                    if lower_bound < x < upper_bound:
                        factor_in_w_signal.remove(x)
                        factor_in_w_signal.append(w_dist.sample())

            factor_in_w_zero = torch.zeros(columns - len(factor_in_w_signal))
            factor_in_w = torch.cat(
                [torch.tensor(factor_in_w_signal), factor_in_w_zero], dim=0
            )

            # weird detour since np.shuffle seems to only locally shuffle
            idx = np.arange(len(factor_in_w))
            np.random.shuffle(idx)
            factor_in_w = factor_in_w[idx]

            w.append(factor_in_w)

    w = torch.stack(w, dim=1)

    # make z
    z = z_dist.sample(sample_shape=[rows, factors])

    # print(f"{w.shape=} {z.shape=}")
    # make y
    y = z @ w.T + eps

    result = {
        "w": w.to(device=device),
        "z": z.to(device=device),
        "eps": eps.to(device=device),
        "y": y.to(device=device),
    }

    return result
