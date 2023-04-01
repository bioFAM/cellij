import json
import os
import pathlib

import click
import muon as mu
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
    """Train on synthetic data."""

    # create output directory recursively
    # does not raise an exception if the directory already exists
    click.echo("Creating output directory...")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    click.echo("Storing arguments...")
    config = locals()
    with open(Path(out_dir).joinpath("config.json"), "w") as f:
        json.dump(config, f)

    data_config = json.load(open(os.path.join(data_dir, "config.json")))
    click.echo(data_config)
    mdata = mu.read_h5mu(os.path.join(data_dir, "data.h5mu"))

    n_feature_groups = len(data_config["n_features"])
    if len(likelihoods) == 0:
        likelihoods = mdata.uns["likelihoods"]

    assert n_feature_groups == len(
        likelihoods
    ), "Number of features and likelihoods must match."

    if n_factors is None:
        n_factors = mdata.obsm["z"].shape[1]

    click.echo(f"Creating instance of `{model}`...")
    model = {"mofa": MOFA}[model](n_factors=n_factors, sparsity_prior=sparsity_prior)
    click.echo("Adding data to model...")
    model.add_data(data=mdata, name="synthetic")
    click.echo("Training model...")
    # TODO: update likelihoods to be a list of likelihoods in the fit method
    # TODO: use random seed to reproduce training
    model.fit(
        likelihood=likelihoods[0].title(),
        epochs=epochs,
        learning_rate=learning_rate,
        verbose_epochs=verbose_epochs,
    )
    click.echo("Saving model...")
    # TODO: save model


if __name__ == "__main__":
    train()
