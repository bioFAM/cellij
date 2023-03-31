import os
import numpy as np
import json
import pathlib
import click

from cellij.core.synthetic import DataGenerator


@click.command()
@click.option("--n-samples", "-n", default=200, help="Number of samples.")
@click.option(
    "--n-features", "-d", default=[400], multiple=True, help="Number of features."
)
@click.option(
    "--likelihoods",
    "-ll",
    default=["normal"],
    multiple=True,
    help="Likelihood for each feature group.",
)
@click.option(
    "--n-factors",
    "-k",
    default=(10, 0, 0),
    nargs=3,
    help="Number of latent factors as a triple: (fully shared, partially shared, private). "
    "Set -k 0 0 0 to generate all possible combinations of latent factors.",
)
@click.option(
    "--factor-sparsity-dist",
    "-fsd",
    default="uniform",
    help="Distribution of the number of active factor loadings.",
)
@click.option(
    "--factor-sparsity-dist-params",
    "-fsdp",
    default=(0.05, 0.15),
    nargs=2,
    help="Parameters for the distribution of the number of active factor loadings for the latent factors.",
)
@click.option(
    "--seed", "-s", default=0, help="Random state to allow for reproducible evaluation."
)
@click.option(
    "--out-dir",
    "-od",
    default="synthetic",
    help="Name of directory to store the generated data.",
)
def generate(
    n_samples,
    n_features,
    n_factors,
    likelihoods,
    factor_sparsity_dist,
    factor_sparsity_dist_params,
    seed,
    out_dir,
):
    """Generate synthetic data."""
    assert len(n_features) == len(
        likelihoods
    ), "Number of features and likelihoods must match."

    # create output directory recursively
    # does not raise an exception if the directory already exists
    click.echo("Creating output directory...")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    click.echo("Storing arguments...")
    config = locals()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f)

    click.echo("Generating synthetic data...")
    dg = DataGenerator(
        n_samples=n_samples,
        n_features=n_features,
        likelihoods=likelihoods,
        n_fully_shared_factors=n_factors[0],
        n_partially_shared_factors=n_factors[1],
        n_private_factors=n_factors[2],
        factor_sparsity_dist=factor_sparsity_dist,
        factor_sparsity_dist_params=factor_sparsity_dist_params,
        seed=seed,
    )
    
    dg.generate(seed=seed, all_feature_group_combs=all([k == 0 for k in n_factors]))
    click.echo("Storing synthetic data...")
    dg.to_mdata().write_h5mu(os.path.join(out_dir, "data.h5mu"))


if __name__ == "__main__":
    generate()
