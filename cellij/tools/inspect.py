from typing import List, Optional, Union
import pyro
import cellij
import numpy as np
from cellij.utils import (
    _get_param_storage_key_prefix
)


def _get_from_param_storage(
    model: cellij.core._factormodel.FactorModel,
    name: str,
    param: str = "locs",
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

    if not isinstance(name, str):
        raise TypeError("Parameter 'name' must be of type str.")

    if not isinstance(param, str):
        raise TypeError("Parameter 'param' must be of type str.")

    if param not in ["locs", "scales"]:
        raise ValueError("Parameter 'param' must be in ['locs', 'scales'].")

    if not isinstance(views, (str, list)):
        raise TypeError("Parameter 'views' must be of type str or list.")

    if isinstance(views, list):
        if not all([isinstance(view, str) for view in views]):
            raise TypeError("Parameter 'views' must be a list of strings.")

    if not isinstance(format, str):
        raise TypeError("Parameter 'format' must be of type str.")

    if format not in ["numpy", "torch", "pyro"]:
        raise ValueError("Parameter 'format' must be in ['numpy', 'torch', 'pyro'].")

    key = _get_param_storage_key_prefix(with_guide=True) + param + "." + name

    if key not in list(model.param_storage.keys()):
        raise ValueError(
            f"Parameter '{key}' not found in parameter storage. Availiable choices are: {list(model.param_storage.keys())}"
        )

    if format == "numpy":
        data = model.param_storage[key].detach().numpy().squeeze()
    elif format == "torch":
        data = model.param_storage[key].detach().squeeze()
    elif format == "pyro":
        data = model.param_storage[key]

    return data


def get_w(
    model: cellij.core._factormodel.FactorModel,
    param: str = "locs",
    views: Union[str, List[str]] = "all",
    format: str = "numpy"
) -> np.ndarray:

    return _get_from_param_storage(
        model=model,
        name="w",
        param=param,
        views=views,
        format=format,
    )


def get_z(
    model: cellij.core._factormodel.FactorModel,
    param: str = "locs",
    views: Union[str, List[str]] = "all",
    format: str = "numpy"
) -> np.ndarray:

    return _get_from_param_storage(
        model=model,
        name="z",
        param=param,
        views=views,
        format=format,
    )
