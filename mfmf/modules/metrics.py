from __future__ import annotations

import matplotlib.pyplot as plt
import mfmf
import pyro
import numpy as np


def plot_elbo(
    model: mfmf.core.FactorModel,
    window: int = 100,
    method: str = "median",
    ax: plt.Axes = None,
):
    """Convenience function to plot the ELBO loss after training a model.

    Args:
        model (mfmf.model.FactorModel): A trained model.
        window (int): The size of the window to apply to smoothing operation on.
        method (str): The smoothing operation, can be ['none', 'median', 'mean'].
        ax (plt.Axes): Optional, an ax to plot the loss onto.

    Return:
        A tuple containing the fig and ax object of the ELBO curve.
    """

    if not isinstance(model, mfmf.core.FactorModel) or not model.is_trained:
        raise TypeError("Only accepting a trained mfmf.model.FactorModel.")

    if not isinstance(window, int):
        raise TypeError("Parameter 'window' must be a positive integer.")
    elif isinstance(window, int) and window <= 0:
        raise ValueError("Parameter 'window' must be a positive integer.")

    if not isinstance(method, str):
        raise TypeError("Parameter 'method' must be one of ['none', 'median', 'mean'].")
    elif method not in ["none", "median", "mean"]:
        raise ValueError(
            "Parameter 'method' must be one of ['none', 'median', 'mean']."
        )

    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("Parameter 'ax' must be a plt.Axes.")

    data = model.loss_during_training.reset_index(drop=True)

    if ax is None:
        _, ax = plt.subplots()

    if method == "none":
        y = data["loss"]
    elif method == "median":
        y = data["loss"].rolling(window).median()
    elif method == "mean":
        y = data["loss"].rolling(window).mean()

    ax.plot(data["epoch"], y, linewidth=2.5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("ELBO")

    return ax


def plot_norm(
    model: mfmf.core.FactorModel,
    key: str,
    ax: plt.Axes = None,
):
    """Convenience function to plot the norm of a parameter after training a model.

    Args:
        model (mfmf.model.FactorModel): A trained model.
        key (str): A pyro parameter key.
        ax (plt.Axes): Optional, an ax to plot the loss onto.

    Return:
        A tuple containing the fig and ax object of the plotted norm.
    """

    if not isinstance(model, mfmf.core.FactorModel) or not model.is_trained:
        raise TypeError("Only accepting a trained mfmf.model.FactorModel.")

    if not isinstance(key, str):
        raise TypeError("Parameter 'key' must be a string.")
    if key not in model.pyro_params.keys():
        raise ValueError(
            "Did not find '"
            + key
            + "' in stored parameters. Available choices are: \n- "
            + "\n- ".join([k for k in model.pyro_params.keys()])
        )

    data = model.pyro_params[key]

    x_vals = [int(k) for k in data.keys()]
    y_vals = [np.linalg.norm(v) for v in data.values()]

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x_vals, y_vals, linewidth=2.5)
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"norm({key})")

    return ax


def render_model(
    model: mfmf.core.FactorModel,
    render_params: bool = True,
    render_distributions: bool = True,
) -> graphviz.graphs.Digraph:
    """Renders the model in the plate representation.

    Args:
        model (mfmf.model.FactorModel): A Pyro model.

    Return:
        model_rendering (graphviz.Digraph): The rendered plate representation of the model.
    """

    if not isinstance(model, mfmf.core.FactorModel) or not model.is_trained:
        raise TypeError("Only accepting a trained mfmf.model.FactorModel.")

    if not isinstance(render_params, bool):
        raise TypeError("Parameter 'render_params' must be True or False.")

    if not isinstance(render_distributions, bool):
        raise TypeError("Parameter 'render_distributions' must be True or False.")

    model_rendering = pyro.render_model(
        model.model,
        # temporarily disabled
        #  render_params=render_params,
        #  render_distributions=render_distributions,
    )

    return model_rendering
