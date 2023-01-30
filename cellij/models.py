import torch

class BaseFactorModel:
    """
    The baseclass from with all cellij models should inherit.

    Attributes
    ----------
    model : cellij.model
        the generative model
    guide : cellij.guide
        the variational distribution
    trainer : cellij.trainer
        defines the training procedure, i.e. the loss function and the optimizer
    dtype : torch.dtype
        the data type of the model 
    device : str, default = str
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
        dtype = torch.float32,
        device = "cpu",
    ):
        
        pass
    
    @property
    def model(self):
        pass
    
    @property.setter
    def model(self, model):
        pass
    
    @property
    def guide(self):
        pass
    
    @property.setter
    def guide(self, guide):
        pass
    
    @property
    def trainer(self):
        pass
    
    @property.setter
    def trainer(self, trainer):
        pass
    
    @property
    def dtype(self):
        pass
    
    @property.setter
    def dtype(self, dtype):
        pass
    
    @property
    def device(self):
        pass
    
    @property.setter
    def device(self, device):
        pass
    
    def add_data(self, name, data, **kwargs):
        pass
    
    def set_data(self, name, data, **kwargs):
        pass
    
    def remove_data(self, name, **kwargs):
        pass
    
    def add_feature_group(self, name, features, **kwargs):
        self._add_group(name=name, features=features, level='feature', **kwargs)
    
    def set_feature_group(self, name, features, **kwargs):
        self._set_group(name=name, features=features, level='feature', **kwargs)
    
    def remove_feature_group(self, name, **kwargs):
        self._remove_group(name=name,  level='feature', **kwargs)
    
    def add_obs_group(self, name, features, **kwargs):
        self._add_group(name=name, features=features, level='obs', **kwargs)
    
    def set_obs_group(self, name, features, **kwargs):
        self._set_group(name=name, features=features, level='obs', **kwargs)
    
    def remove_obs_group(self, name, **kwargs):
        self._remove_group(name=name,  level='obs', **kwargs)
    
    def _add_group(self, name, group, level, **kwargs):
        pass
    
    def _set_group(self, name, group, level, **kwargs):
        pass
    
    def _remove_group(self, name, level, **kwargs):
        pass
    
    def fit(self, dry_run=False, **kwargs):
        pass
    
    
    