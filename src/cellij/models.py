from typing import Dict, Optional, Union

import pyro
import torch
from pyro.nn import PyroModule


class Model(PyroModule):
    def __init__(self, device: Optional[Union[torch.device, int]] = None):
        """Generative model base class.

        The class uses a fixed number of tensor dimensions, associated with
        - -1: samples
        - -2: factors
        - -3: features
        - -4: views
        - -5: groups

        Parameters
        ----------
        device : Optional[Union[torch.device, int]], optional
            Device to run the model on, by default None
        """
        super().__init__(name="Model")
        self.device = device

    def get_plates(self, *args, **kwargs) -> Dict[str, pyro.plate]:
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError()
