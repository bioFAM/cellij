from typing import List, Optional, Union
import pyro
import cellij
import numpy as np
import numpy.typing as npt


def pull_from_param_storage(
    model: cellij.core._factormodel.FactorModel,
    name: str,
    views: Union[str, List[str]] = "all",
    format: str = "numpy",
) -> np.ndarray:
    """Pulls a parameter from the pyro parameter storage.

    Parameters
    ----------
    name : str
        The name of the parameter to be pulled.
    format : str
        The format in which the parameter should be returned.
        Options are: "numpy", "torch", "pyro".

    Returns
    -------
    parameter : torch.Tensor or numpy.ndarray
        The parameter pulled from the pyro parameter storage.
    """
    if not isinstance(model, cellij.core._factormodel.FactorModel):
        raise TypeError(
            "Parameter 'model' must be of type cellij.core._factormodel.FactorModel."
        )

    if not isinstance(name, str):
        raise TypeError("Parameter 'name' must be of type str.")

    if not isinstance(format, str):
        raise TypeError("Parameter 'format' must be of type str.")

    if not isinstance(views, (str, list)):
        raise TypeError("Parameter 'views' must be of type str or list.")

    if isinstance(views, list):
        if not all([isinstance(view, str) for view in views]):
            raise TypeError("Parameter 'views' must be a list of strings.")

    if format not in ["numpy", "torch", "pyro"]:
        raise ValueError("Parameter 'format' must be in ['numpy', 'torch', 'pyro'].")

    param_storage = model._model.params

    if name not in param_storage.keys():
        raise ValueError(
            f"Parameter '{name}' not found in parameter storage. Availiable choices are: {param_storage.keys()}"
        )

    if format == "numpy":
        data = param_storage[name].detach().numpy().squeeze()
    elif format == "torch":
        data = param_storage[name].detach().squeeze()
    elif format == "pyro":
        data = param_storage[name]

    if name == "w":
        if views == "all":
            return data
        else:
            if isinstance(views, str):
                views = [views]
            return data[:, views, :]

    return data


def get_w(
    model: cellij.core._factormodel.FactorModel, format: str = "numpy"
) -> np.ndarray:

    return pull_from_param_storage(model, "w", format)


def get_z(
    model: cellij.core._factormodel.FactorModel, format: str = "numpy"
) -> np.ndarray:

    return pull_from_param_storage(model, "z", format)
