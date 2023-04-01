from typing import List, Optional, Union
import pyro
import cellij
import numpy as np
from cellij.utils import _get_param_storage_key_prefix


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
        Options are: "numpy", "torch".

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

    if format not in ["numpy", "torch"]:
        raise ValueError("Parameter 'format' must be in ['numpy', 'torch'].")

    key = _get_param_storage_key_prefix(with_guide=True) + param + "." + name

    if key not in list(model.param_storage.keys()):
        raise ValueError(
            f"Parameter '{key}' not found in parameter storage. Availiable choices are: {', '.join(list(model.param_storage.keys()))}"
        )

    data = model.param_storage[key]

    if views != "all":

        if isinstance(views, str):

            if views not in model.data._names:
                raise ValueError(
                    f"Parameter 'views' must be in {list(model.data._names)}."
                )

            if format == "numpy":
                result = data[..., model.data._feature_idx[views]].numpy()
            elif format == "torch":
                result = data[..., model.data._feature_idx[views]]

            return result

        elif isinstance(views, list):

            if not all([view in model.data._names for view in views]):
                raise ValueError(
                    f"All elements in 'views' must be in {list(model.data._names)}."
                )

            result = {}
            for view in views:
                if format == "numpy":
                    result[view] = data[..., model.data._feature_idx[view]].numpy()
                elif format == "torch":
                    result[view] = data[..., model.data._feature_idx[view]]

            return result

    elif views == "all":

        return data


def _get_w(
    model: cellij.core._factormodel.FactorModel,
    param: str = "locs",
    views: Union[str, List[str]] = "all",
    format: str = "numpy",
) -> np.ndarray:

    return _get_from_param_storage(
        model=model,
        name="w",
        param=param,
        views=views,
        format=format,
    )


def _get_z(
    model: cellij.core._factormodel.FactorModel,
    param: str = "locs",
    views: Union[str, List[str]] = "all",
    format: str = "numpy",
) -> np.ndarray:

    return _get_from_param_storage(
        model=model,
        name="z",
        param=param,
        views=views,
        format=format,
    )
