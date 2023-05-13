import json
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from addict import Dict
import mudata

from cellij.core.models import MOFA
from cellij.core.synthetic import DataGenerator
from cellij.utils import load_model, set_all_seeds

OVERWRITE = False
if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    CUDA = torch.cuda.is_available()
    torch.cuda.set_device(1)
    device = torch.device("cuda")
else:
    torch.set_default_tensor_type("torch.FloatTensor")
    device = torch.device("cpu")

perf_timer = Dict()

N_SHARED_FACTORS = 20
N_PARTIAL_FACTORS = 0
N_PRIVATE_FACTORS = 0
N_SAMPLES = 200
MISSINGS = 0.0
N_FACTORS_ESTIMATED = 20

PATH = "/data/m015k/data/cellij/benchmark/benchmark_v1_features/"

for seed in [0, 1, 2, 3, 4]:
    set_all_seeds(seed)

    for grid_features in [50, 100, 200, 400, 800, 1000, 2000, 3500, 5000]:
        n_samples = N_SAMPLES
        n_features = [grid_features, grid_features, grid_features]
        n_views = len(n_features)
        likelihoods = ["Normal" for _ in range(n_views)]

        n_fully_shared_factors = N_SHARED_FACTORS
        n_partially_shared_factors = N_PARTIAL_FACTORS
        n_private_factors = N_PRIVATE_FACTORS

        n_covariates = 0

        if not (
            Path(PATH)
            .joinpath(
                f"dgp_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
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
                Path(PATH).joinpath(
                    f"dgp_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                )
            )
            print("Saved data...")
        else:
            print("Loading data...")
            mdata = mudata.read(
                Path(PATH).joinpath(
                    f"dgp_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{seed}.h5mu"
                )
            )

        for lr in [0.1, 0.01, 0.001, 0.0001]:
            for sparsity_prior, prior_params in [
                # ("Nonnegativity", {}),
                # ("SpikeNSlab", {"relaxed_bernoulli": True, "temperature": 0.1}),
                ("SpikeNSlab", {"relaxed_bernoulli": False}),
                ("Horseshoe", {"tau_scale": 1.0, "lambda_scale": 1.0}),
                ("Lasso", {"lasso_scale": 0.1}),
            ]:
                s_params = (
                    "_".join([f"{k}={v}" for k, v in prior_params.items()])
                    if prior_params
                    else ""
                )
                filename = Path(PATH).joinpath(
                    f"model_v1_features_{N_SHARED_FACTORS}_{N_PARTIAL_FACTORS}_{N_PRIVATE_FACTORS}_{N_SAMPLES}_{grid_features}_{MISSINGS}_{sparsity_prior}_{N_FACTORS_ESTIMATED}_{lr}_{seed}_{s_params}.pkl"
                )

                if Path(filename).exists() and not OVERWRITE:
                    print(f"Loading {filename}")
                    model = load_model(str(filename))
                else:
                    model = MOFA(
                        n_factors=N_FACTORS_ESTIMATED,
                        sparsity_prior=sparsity_prior,
                        **prior_params,
                    )
                    model.add_data(data=mdata, na_strategy=None)
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
                    model.save(filename=str(filename), overwrite=True)
                    perf_timer[str(filename).replace(".pkl", "")] = end - start

                    with open(Path(PATH).joinpath("perf_timer.json"), "w") as fp:
                        json.dump(perf_timer, fp)
