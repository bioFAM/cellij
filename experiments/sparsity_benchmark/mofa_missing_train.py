import json
from pathlib import Path
from timeit import default_timer as timer

import mudata
import numpy as np
import torch
from addict import Dict

from cellij.core.models import MOFA
from cellij.core.synthetic import DataGenerator
from cellij.utils import load_model, set_all_seeds
import scanpy as sc


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
# N_FACTORS_ESTIMATED = 20
OVERWRITE = False

# 20 Factors estimated for full grid  @ 200 samples
# 60 - 5 factors estimated for 400 and 1000 features @ 200 samples

PATH_DGP = "/home/m015k/code/cellij/experiments/sparsity_benchmark/data/"
PATH_MODELS = "/data/m015k/data/cellij/benchmark/benchmark_v1_missings/"

for seed in [0]:
    set_all_seeds(seed)

    for lr in [0.01]:
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

                if not (
                    Path(PATH_DGP)
                    .joinpath(
                        f"dgp_missing_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                    )
                    .exists()
                ):
                    print("Creating data...")

                    dg = DataGenerator(
                        n_samples,
                        n_features,
                        likelihoods,
                        n_fully_shared_factors,
                        n_partially_shared_factors,
                        n_private_factors,
                        n_covariates=n_covariates,
                    )

                    rng = dg.generate(seed=seed)
                    dg.normalize(with_std=False)
                    feature_offsets = [0] + np.cumsum(n_features).tolist()
                    vlines = feature_offsets[1:-1]
                    mdata = dg.to_mdata()
                    mdata.write(
                        Path(PATH_DGP).joinpath(
                            f"dgp_missing_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                        )
                    )

                    # Save list of random missings
                    for percentage in [0, 0.05, 0.1, 0.2, 0.5, 0.9]:
                        print("- Saving data for percentage:{percentage}...")
                        for view in range(3):
                            num_elements = N_SAMPLES * grid_features
                            num_values_to_set_missing = int(percentage * num_elements)

                            # Generate random indices to set as missing
                            random_indices = np.random.choice(
                                np.arange(num_elements),
                                size=num_values_to_set_missing,
                                replace=False,
                            )
                            # Save random_indices to file
                            np.save(
                                Path(PATH_DGP).joinpath(
                                    f"random_idx_missing_{view}_{percentage}_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.npy"
                                ),
                                random_indices,
                            )

                    print("Saved data...")
                else:
                    print(f"Loading data from {PATH_DGP}...")
                    mdata = mudata.read(
                        Path(PATH_DGP).joinpath(
                            f"dgp_missing_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                        )
                    )

                for missing_percentage in [0, 0.05, 0.1, 0.2, 0.5, 0.9]:
                    # Apply missings to data
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
                        (
                            "SpikeNSlab",
                            {"relaxed_bernoulli": True, "temperature": 0.1},
                        ),
                        (
                            "SpikeNSlab",
                            {"relaxed_bernoulli": True, "temperature": 0.01},
                        ),
                        ("SpikeNSlab", {"relaxed_bernoulli": False}),
                        ("Lasso", {"lasso_scale": 0.1}),
                        ("Horseshoe", {"tau_scale": 1.0, "lambda_scale": 1.0}),
                        ("Horseshoe", {"tau_scale": 0.1, "lambda_scale": 1.0}),
                        ("HorseshoePlus", {"tau_const": 0.1, "eta_scale": 1.0}),
                        (
                            "Horseshoe",
                            {
                                "tau_scale": 0.1,
                                "lambda_scale": 1.0,
                                "delta_tau": True,
                            },
                        ),
                        (
                            "Horseshoe",
                            {
                                "tau_scale": 1.0,
                                "lambda_scale": 1.0,
                                "regularized": True,
                            },
                        ),
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
                            f" - {sparsity_prior} | {prior_params} | {lr} | {grid_features} | {N_FACTORS_ESTIMATED} | {seed} | {missing_percentage}"
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

                        if Path(filename).exists() and not OVERWRITE:
                            print(f"Loading {filename}")
                            model = load_model(str(filename))
                        else:
                            print("- Using missings: {missing_percentage}...")
                            for name, anndata in mdata.mod.items():
                                # Print number of missings
                                print(
                                    f" - {name}: {np.isnan(anndata.X).sum()} missings"
                                )

                            model = MOFA(
                                n_factors=N_FACTORS_ESTIMATED,
                                sparsity_prior=sparsity_prior,
                                **prior_params,
                            )
                            model.add_data(data=mdata)
                            perf_timer = (
                                {}
                            )  # Measure time until convergence for each model
                            start = timer()
                            losses = model.fit(
                                likelihoods=mdata.uns["likelihoods"],
                                epochs=25_000,
                                num_particles=20,
                                learning_rate=lr,
                                verbose_epochs=500,
                                min_delta=0.01,
                            )
                            end = timer()
                            model.save(filename=str(filename), overwrite=OVERWRITE)
                            perf_timer[str(filename).replace(".pkl", "")] = end - start

                            with open(
                                str(filename).replace(".pkl", "_perf_timer.json"),
                                "w",
                            ) as fp:
                                json.dump(perf_timer, fp)
