import json
from pathlib import Path
from timeit import default_timer as timer

import mudata
import numpy as np
import torch
from addict import Dict
import cellij
from functools import reduce
import anndata as ad
import muon as mu

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

PATH_DGP = "/data/m015k/data/cellij/benchmark/benchmark_cellij_missing/"
PATH_MODELS = "/data/m015k/data/cellij/benchmark/benchmark_cellij_missing/"

# Import CLL loader from Cellij
imp = cellij.core.Importer()

for seed in [0, 1, 2, 3, 4]:
    set_all_seeds(seed)

    for n_factors_estimated in [10]:  # 25
        for lr in [1e-3]:
            for percent_missing in [
                # 0.02,
                0.1,
                # 0.2,
                0.3,
                # 0.4,
                0.5,
                # 0.6,
                0.7,
                # 0.8,
                0.9,
                # 0.95,
            ]:
                # Load a fresh dataset
                mdata = imp.load_CLL()
                
                # Loop over all modalities and replace x percent of the values with missings
                for modality, anndata in mdata.mod.items():
                    print(modality)
                    # Load random indices if exists
                    filename_data = Path(PATH_DGP).joinpath(
                        f"model_cll_random_indices_{seed}_{percent_missing}_{modality}.npy"
                    )
                    if filename_data.exists():
                        print(f"Loading random indices from {filename_data}")
                        random_indices = np.load(filename_data)
                    else:
                        n_features, n_samples = anndata.shape[0], anndata.shape[1]
                        num_elements = n_features * n_samples

                        already_missing_indices = np.where(
                            np.isnan(anndata.X.flatten())
                        )[0]
                        values_potential_missing_indices = list(
                            set(np.arange(num_elements)) - set(already_missing_indices)
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

                    print("Missings before: ", np.sum(np.isnan(anndata.X)))
                    anndata_flattened = anndata.X.flatten()
                    anndata_flattened[random_indices] = np.nan
                    mdata[modality].X = anndata_flattened.reshape(anndata.shape)
                    print("Missings after: ", np.sum(np.isnan(anndata.X)))

                    # Center features at zero mean
                    mdata[modality].X = mdata[modality].X - np.nanmean(
                        mdata[modality].X, axis=0
                    )

                # from functools import reduce
                # # Use the intersection of observations
                # obs_names = mdata.obs.index.values
                # common_obs = reduce(np.intersect1d, [v.obs_names.values for k, v in mdata.mod.items()])
                # mdata = mdata[common_obs]
                
                # # Loop over all modalities and drop missing rows
                # mdata = imp.load_CLL()
                # non_nan_idx = set()
                # for modality, anndata in mdata.mod.items():
                #     print(modality)
                #     non_nan_idx |= set(anndata[np.isnan(anndata.X).all(axis=1)].obs.index)
                #     print(len(non_nan_idx))

                # import pandas as pd
                # for modality, anndata in mdata.mod.items():
                #     print(anndata.X.shape, pd.DataFrame(anndata.X).dropna(how='all').shape)

                # Plot a histogram of each modality
                for modality, anndata in mdata.mod.items():
                    import matplotlib.pyplot as plt

                    plt.hist(anndata.X.flatten(), bins=100)
                    plt.title(
                        f"{modality} | {anndata.shape} | {np.sum(np.isnan(anndata.X))}"
                    )
                    plt.show()

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
                    # (
                    #     "Horseshoe",
                    #     {
                    #         "tau_scale": 0.1,
                    #         "lambda_scale": 1.0,
                    #         "theta_scale": 1.0,
                    #         "delta_tau": True,
                    #         "regularized": False,
                    #         "ard": False,
                    #     },
                    # ),
                    # (
                    #     "Horseshoe",
                    #     {
                    #         "tau_scale": 0.1,
                    #         "lambda_scale": 1.0,
                    #         "theta_scale": 1.0,
                    #         "delta_tau": False,
                    #         "regularized": True,
                    #         "ard": False,
                    #     },
                    # ),
                    # (
                    #     "Horseshoe",
                    #     {
                    #         "tau_scale": 0.1,
                    #         "lambda_scale": 1.0,
                    #         "theta_scale": 1.0,
                    #         "delta_tau": False,
                    #         "regularized": False,
                    #         "ard": True,
                    #     },
                    # ),
                    # (
                    #     "Horseshoe",
                    #     {
                    #         "tau_scale": 0.1,
                    #         "lambda_scale": 1.0,
                    #         "theta_scale": 1.0,
                    #         "delta_tau": False,
                    #         "regularized": True,
                    #         "ard": True,
                    #     },
                    # ),
                    # ("SpikeAndSlab", {"relaxed_bernoulli": True, "temperature": 0.1}),
                    ("SpikeAndSlab", {"relaxed_bernoulli": False}),
                    # (
                    #     "SpikeAndSlabLasso",
                    #     {
                    #         "lambda_spike": 20.0,
                    #         "lambda_slab": 0.01,
                    #         "relaxed_bernoulli": True,
                    #         "temperature": 0.1,
                    #     },
                    # ),
                ]:
                    print(
                        f" - {sparsity_prior} | {prior_params} | {lr} | {n_factors_estimated} | {seed}"
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
                        model = MOFA(
                            n_factors=n_factors_estimated,
                            sparsity_prior=sparsity_prior,
                            **prior_params,
                        )
                        model.add_data(data=mdata, na_strategy=None)
                        perf_timer = {}  # Measure time until convergence for each model
                        start = timer()
                        losses = model.fit(
                            likelihoods={
                                "drugs": "Gaussian",
                                "methylation": "Gaussian",
                                "mrna": "Gaussian",
                                "mutations": "Bernoulli",
                            },
                            epochs=30_000,
                            num_particles=20,
                            learning_rate=lr,
                            verbose_epochs=500,
                            min_delta=0.01,
                        )
                        end = timer()
                        model.save(filename=str(filename), overwrite=False)
                        perf_timer[str(filename).replace(".pkl", "")] = end - start

                        with open(
                            str(filename).replace(".pkl", "_perf_timer.json"), "w"
                        ) as fp:
                            json.dump(perf_timer, fp)
