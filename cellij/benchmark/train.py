import json
from pathlib import Path

import click
import muon as mu
import numpy as np
import pyro
import torch

from cellij.core.models import MOFA


@click.command()
@click.option(
    "--data-dir",
    "-dd",
    default="synthetic",
    help="Directory from where to load the data.",
)
@click.option("--model", "-m", default="mofa", help="Model to train.")
@click.option(
    "--n-factors",
    "-k",
    help="Number of latent factors, inferred if not provided.",
)
@click.option(
    "--sparsity-prior",
    "-sp",
    default="spikeandslab",
    help="Sparsity prior for the factor loadings.",
)
@click.option(
    "--likelihoods",
    "-ll",
    required=False,
    multiple=True,
    help="Likelihood for each feature group, inferred if not provided.",
)
@click.option(
    "--epochs",
    "-e",
    default=10000,
    help="Number of epochs to train for.",
)
@click.option(
    "--learning-rate",
    "-lr",
    default=0.003,
    help="Learning rate for the optimizer.",
)
@click.option(
    "--verbose-epochs",
    "-ve",
    default=100,
    help="Frequency of printing the loss during training.",
)
@click.option(
    "--seed", "-s", default=0, help="Random state to allow for reproducible evaluation."
)
@click.option(
    "--out-dir",
    "-od",
    default="training_output",
    help="Name of directory to store the training output.",
)
def train(
    data_dir,
    model,
    n_factors,
    sparsity_prior,
    likelihoods,
    epochs,
    learning_rate,
    verbose_epochs,
    seed,
    out_dir,
):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    """Train on synthetic data."""
    if isinstance(n_factors, str):
        n_factors = int(n_factors)

    # create output directory recursively
    # does not raise an exception if the directory already exists
    click.echo("Creating output directory...")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    click.echo("Storing arguments...")
    config = locals()
    with open(Path(out_dir).joinpath("config.json"), "w") as f:
        json.dump(config, f)

    click.echo(f"Loading data from `{data_dir}`...")
    with open(Path(data_dir).joinpath("config.json"), "r") as f:
        data_config = json.load(f)

    click.echo(data_config)
    mdata = mu.read_h5mu(Path(data_dir).joinpath("data.h5mu"))

    n_feature_groups = len(data_config["n_features"])
    if len(likelihoods) == 0:
        likelihoods = mdata.uns["likelihoods"]

    if n_feature_groups != len(likelihoods):
        raise ValueError(
            "Number of feature groups and likelihoods must be equal. "
            f"Got {n_feature_groups} features and {len(likelihoods)} likelihoods."
        )

    if n_factors is None:
        n_factors = mdata.obsm["z"].shape[1]

    click.echo(f"Creating instance of `{model}`...")
    model = {"mofa": MOFA}[model](n_factors=n_factors, sparsity_prior=sparsity_prior)
    click.echo("Adding data to model...")
    model.add_data(data=mdata, name="synthetic")
    click.echo("Training model...")
    # TODO: update likelihoods to be a list of likelihoods in the fit method
    # TODO: use random seed to reproduce training
    # TODO: Expose more paraemters to the user, e.g., autoguide, etc.
    model.fit(
        # TODO: use likelihoods parameter instead of inferring from mdata
        likelihoods=mdata.uns["likelihoods"],
        epochs=epochs,
        learning_rate=learning_rate,
        verbose_epochs=verbose_epochs,
    )
    return model
    click.echo("Saving model...")
    # TODO: save model
    np.save(Path(out_dir).joinpath("z.npy"), get_z())
    np.save(Path(out_dir).joinpath("w.npy"), get_w(sparsity_prior))
    click.echo(f"Model parameters saved in `{out_dir}`...")
    return model


def get_w(sparsity_prior):
    if sparsity_prior == "Horseshoe":
        unscaled_w = pyro.get_param_store()["FactorModel._guide.locs.unscaled_w"]
        w_scale = pyro.get_param_store()["FactorModel._guide.locs.w_scale"]
        w = unscaled_w * w_scale
    elif sparsity_prior == "Spikeandslab-Lasso":
        samples_bernoulli = torch.cat(
            [
                pyro.get_param_store()[
                    f"FactorModel._guide.locs.samples_bernoulli_{idx}"
                ]
                for idx in range(3)
            ],
            dim=1,
        )
        samples_lambda_spike = torch.cat(
            [
                pyro.get_param_store()[
                    f"FactorModel._guide.locs.samples_lambda_spike_{idx}"
                ]
                for idx in range(3)
            ],
            dim=1,
        )
        samples_lambda_slab = torch.cat(
            [
                pyro.get_param_store()[
                    f"FactorModel._guide.locs.samples_lambda_slab_{idx}"
                ]
                for idx in range(3)
            ],
            dim=1,
        )
        w = (
            1 - samples_bernoulli
        ) * samples_lambda_spike + samples_bernoulli * samples_lambda_slab
    else:
        w = pyro.get_param_store()["FactorModel._guide.locs.w"]
    return w.squeeze().cpu().detach().numpy()


def get_z():
    return (
        pyro.get_param_store()
        .get_param("FactorModel._guide.locs.z")
        .squeeze()
        .cpu()
        .detach()
        .numpy()
    )


if __name__ == "__main__":
    train()
