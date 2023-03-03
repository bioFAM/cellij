import pyro
import numpy as np
import torch
from pyro.infer import SVI
from pyro.nn import PyroModule
import pandas as pd

from cellij.core._data import DataContainer


class FactorModel(PyroModule):
    """Base class for all estimators in cellij.

    Attributes
    ----------
    model : cellij.model
        The generative model
    guide : Union[str, pyro.infer.autoguide.initialization, cellij.guide]
        The variational distribution
    n_factors : int
        The number of factors
    trainer : cellij.trainer
        Defines the training procedure, i.e. the loss function and the optimizer
    dtype : torch.dtype, default = torch.float32
        The data type of the model
    device : str, default = "cpu"
        The device on which the model is run


    Methods
    -------
    add_data(name, data, **kwargs)
        Adds data to the model
    set_data(name, data, **kwargs)
        Overwrites data with the same name
    remove_data(name, **kwargs)
        Removes a data from the model by its name

    add_feature_group(name, features, **kwargs)
        Delegates to _add_group(..., level = 'feature')
    set_feature_group(name, features, **kwargs)
        Delegates to _set_group(..., level = 'feature')
    remove_feature_group(name, **kwargs)
        Delegates to _remove_group(..., level = 'feature')

    add_obs_group(name, features, **kwargs)
        Delegates to _add_group(..., level = 'obs')
    set_obs_group(name, features, **kwargs)
        Delegates to _set_group(..., level = 'obs')
    remove_obs_group(name, **kwargs)
        Delegates to _remove_group(..., level = 'obs')

    _add_group(name, group, level, **kwargs)
        Adds a group to the model
    _set_group(name, group, level, **kwargs)
        Overwrites a group with the same name
    _remove_group(name, level, **kwargs)
        Removes a group from the model by its name

    fit(dry_run=False, **kwargs)

    """

    def __init__(self, model, guide, n_factors, trainer, dtype=torch.float32, device="cpu", sparsity_prior=None):
        super().__init__(name="FactorModel")

        self._model = model
        self._n_factors = n_factors
        self._trainer = trainer
        self._dtype = dtype
        self._device = device
        self._data = DataContainer()
        self._is_trained = False
        self._feature_groups = {}
        self._obs_groups = {}
        self.sparsity_prior = sparsity_prior

        # Setup
        if isinstance(guide, str):
            if guide == "AutoNormal":
                self._guide = pyro.infer.autoguide.AutoNormal(self._model)
            else:
                raise ValueError(f"Unknown guide: {guide}")

    @property
    def model(self):
        pass

    @model.setter
    def model(self, model):
        pass

    @property
    def guide(self):
        pass

    @guide.setter
    def guide(self, guide):
        self._guide = guide

    @property
    def n_factors(self):
        return self._n_factors

    @n_factors.setter
    def n_factors(self, n_factors):
        self._n_factors = n_factors

    @property
    def trainer(self):
        pass

    @trainer.setter
    def trainer(self, trainer):
        pass

    @property
    def dtype(self):
        pass

    @dtype.setter
    def dtype(self, dtype):
        pass

    @property
    def device(self):
        pass

    @device.setter
    def device(self, device):
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, *args):
        raise AttributeError("Use `add_data()`, `set_data` or `remove_data()` to modify this property.")

    @property
    def is_trained(self):
        return self._is_trained

    @is_trained.setter
    def is_trained(self, *args):
        raise AttributeError("This property is read-only.")

    @property
    def feature_groups(self):
        return self._feature_groups

    @feature_groups.setter
    def feature_groups(self, *args):
        raise AttributeError(
            "Use `add_feature_group()`, `set_feature_group` or `remove_feature_group()` to modify this property."
        )

    @property
    def obs_groups(self):
        return self._obs_groups

    @obs_groups.setter
    def obs_groups(self, *args):
        raise AttributeError("Use `add_obs_group()`, `set_obs_group` or `remove_obs_group()` to modify this property.")

    def add_data(self, data, **kwargs):
        # take in any form of tabular data
        # if it is anything but anndata or mudata, convert it to anndata
        # if it is anndata, add it to self.data
        # if it is mudata, add the individual anndata to self.data

        self._data = data

    def set_data(self, name, data, **kwargs):
        pass

    def remove_data(self, name, **kwargs):
        pass

    def add_feature_group(self, name, features, **kwargs):
        self._add_group(name=name, features=features, level="feature", **kwargs)

    def set_feature_group(self, name, features, **kwargs):
        self._set_group(name=name, features=features, level="feature", **kwargs)

    def remove_feature_group(self, name, **kwargs):
        self._remove_group(name=name, level="feature", **kwargs)

    def add_obs_group(self, name, features, **kwargs):
        self._add_group(name=name, features=features, level="obs", **kwargs)

    def set_obs_group(self, name, features, **kwargs):
        self._set_group(name=name, features=features, level="obs", **kwargs)

    def remove_obs_group(self, name, **kwargs):
        self._remove_group(name=name, level="obs", **kwargs)

    def _add_group(self, name, group, level, **kwargs):
        if name in self._feature_groups.keys() or name in self._obs_groups.keys():
            raise ValueError(f"A group with the name {name} already exists.")

        if level == "feature":
            self._feature_groups[name] = group
        elif level == "obs":
            self._obs_groups[name] = group
        else:
            raise ValueError(f"Level must be 'feature' or 'obs', not {level}")

    def _set_group(self, name, group, level, **kwargs):
        if level == "feature":
            self._feature_groups[name] = group
        elif level == "obs":
            self._obs_groups[name] = group
        else:
            raise ValueError(f"Level must be 'feature' or 'obs', not {level}")

    def _remove_group(self, name, level, **kwargs):
        if name not in self._feature_groups.keys() and name not in self._obs_groups.keys():
            raise ValueError(f"No group with the name {name} exists.")

        if level == "feature":
            del self._feature_groups[name]
        elif level == "obs":
            del self._obs_groups[name]
        else:
            raise ValueError(f"Level must be 'feature' or 'obs', not {level}")

    def fit(self, likelihood, epochs=1000, verbose_epochs=100):
        # Clear pyro param
        pyro.clear_param_store()

        # Check if data is set
        if self._data is None:
            raise ValueError("No data set.")

        # Provide data information to generative model
        n_obs = self._data.n_obs
        feature_offsets = [0] + list(np.cumsum([v.shape[1] for v in self._data.mod.values()]))
        self._model._setup(n_obs=n_obs, feature_offsets=feature_offsets, sparsity_prior=self.sparsity_prior)
        
        if likelihood != "Normal":
            raise ValueError("Only Normal likelihood is implemented so far.")

        svi = SVI(
            model=self._model,
            guide=self._guide,
            optim=pyro.optim.Adam({"lr": 0.005, "betas": (0.95, 0.999)}),
            loss=pyro.infer.Trace_ELBO(),
        )

        # Concatenate all values in self._data.mod.values()
        cat_data = torch.tensor(pd.concat([v.to_df() for v in self._data.mod.values()], axis=1).to_numpy())
        # Center data
        cat_data = cat_data - cat_data.mean(dim=0)
        # Impute missing values
        cat_data = cat_data.fill_(0)

        for i in range(epochs + 1):
            loss = svi.step(X=cat_data)

            if i % verbose_epochs == 0:
                print(f"Epoch {i:>6}: {loss:>14.2f}")

        self._is_trained = True
