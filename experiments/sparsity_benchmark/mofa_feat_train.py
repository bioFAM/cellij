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
OVERWRITE = False

# 20 Factors estimated for full grid  @ 200 samples
# 60 - 5 factors estimated for 400 and 1000 features @ 200 samples

PATH_DGP = "/home/m015k/code/cellij/experiments/sparsity_benchmark/data/"
PATH_MODELS = "/data/m015k/data/cellij/benchmark/benchmark_v2_features/"

for seed in [0, 1, 2]:
    set_all_seeds(seed)

    for lr in reversed([0.01, 0.1, 0.001]):
        for grid_features in [10000]:
            for n_factors_estimated in [20]:
                n_samples = N_SAMPLES
                n_features = [grid_features, grid_features, grid_features]
                n_views = len(n_features)
                likelihoods = ["Normal" for _ in range(n_views)]

                n_fully_shared_factors = N_SHARED_FACTORS
                n_partially_shared_factors = N_PARTIAL_FACTORS
                n_private_factors = N_PRIVATE_FACTORS
                n_covariates = 0

                data_path = (
                    Path(PATH_DGP)
                    / f"dgp_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                )
                if not data_path.exists():
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
                    mdata.write(str(data_path))
                    print("Saved data...")
                else:
                    print(f"Loading data from {PATH_DGP}...")
                    mdata = mudata.read(str(data_path))

                for sparsity_prior, prior_params in [
                    # (None, {}),
                    # ("Lasso", {"lasso_scale": 0.1}),
                    # (
                    #     "Horseshoe",
                    #     {
                    #         "tau_scale": 0.1,
                    #         "lambda_scale": 1.0,
                    #         "theta_scale": 1.0,
                    #         "delta_tau": False,
                    #         "regularized": False,
                    #         "ard": False,
                    #     },
                    # ),
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
                        f" - {sparsity_prior} | {prior_params} | {lr} | {grid_features} | {n_factors_estimated} | {seed}"
                    )
                    # Combine all parameters used for the prior into a string
                    # This allows to train model with the same prior but different parameters
                    s_params = (
                        "_".join([f"{k}={v}" for k, v in prior_params.items()])
                        if prior_params
                        else "None"
                    )
                    filename = Path(PATH_MODELS).joinpath(
                        f"model_v1_features_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{sparsity_prior}_{n_factors_estimated}_{lr}_{seed}_{s_params}.pkl"
                    )

                    if Path(filename).exists() and not OVERWRITE:
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
                            likelihoods=mdata.uns["likelihoods"],
                            epochs=30_000,
                            num_particles=20,
                            learning_rate=lr,
                            verbose_epochs=500,
                            min_delta=0.01,
                        )
                        end = timer()
                        model.save(filename=str(filename), overwrite=OVERWRITE)
                        perf_timer[str(filename).replace(".pkl", "")] = end - start

                        with open(
                            str(filename).replace(".pkl", "_perf_timer.json"), "w"
                        ) as fp:
                            json.dump(perf_timer, fp)
