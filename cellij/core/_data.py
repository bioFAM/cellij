from collections import UserDict

class DataContainer(UserDict):
    """Container to hold all data for a FactorModel.
    
    Holds a list of anndata objects, and provides methods to add, remove, and
    modify data. Also provides methods to merge those anndata and return them
    as a tensor for training or a mudata for the user.
        
    """
    
    def to_mudata(self):
        
        """Returns a mudata object holding all anndata added to the container."""
        
        pass
    
    def prepare_for_training(self):
        
        """Merges all data and converts it to a tensor which is used during training."""
        
        pass