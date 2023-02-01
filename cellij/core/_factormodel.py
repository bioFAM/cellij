import torch
from cellij.core._data import DataContainer
from cellij.core._group import Group


class FactorModel:
    """Base class for all estimators in cellij.

    Attributes
    ----------
    model : cellij.model
        the generative model
    guide : cellij.guide
        the variational distribution
    trainer : cellij.trainer
        defines the training procedure, i.e. the loss function and the optimizer
    dtype : torch.dtype, default = torch.float32
        the data type of the model
    device : str, default = "cpu"
        the device on which the model is run


    Methods
    -------
    add_data(name, data, **kwargs)
        adds data to the model
    set_data(name, data, **kwargs)
        overwrites data with the same name
    remove_data(name, **kwargs)
        removes a data from the model by its name

    add_feature_group(name, features, **kwargs)
        delegates to _add_group(..., level = 'feature')
    set_feature_group(name, features, **kwargs)
        delegates to _set_group(..., level = 'feature')
    remove_feature_group(name, **kwargs)
        delegates to _remove_group(..., level = 'feature')

    add_obs_group(name, features, **kwargs)
        delegates to _add_group(..., level = 'obs')
    set_obs_group(name, features, **kwargs)
        delegates to _set_group(..., level = 'obs')
    remove_obs_group(name, **kwargs)
        delegates to _remove_group(..., level = 'obs')

    _add_group(name, group, level, **kwargs)
        adds a group to the model
    _set_group(name, group, level, **kwargs)
        overwrites a group with the same name
    _remove_group(name, level, **kwargs)
        removes a group from the model by its name

    fit(dry_run=False, **kwargs)

    """

    def __init__(
        self,
        model,
        guide,
        trainer,
        dtype=torch.float32,
        device="cpu",
    ):

        self._model = model
        self._guide = guide
        self._trainer = trainer
        self._dtype = dtype
        self._device = device
        self._data = DataContainer()
        self._is_trained = False
        self._feature_groups = {}
        self._obs_groups = {}

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
        pass

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

    def add_data(self, name, data, **kwargs):

        # take in any form of tabular data
        # if it is anything but anndata or mudata, convert it to anndata
        # if it is anndata, add it to self.data
        # if it is mudata, add the individual anndata to self.data

        pass

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

    def fit(self, dry_run=False, **kwargs):
        pass
