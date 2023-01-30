from typing import Optional, Union

import torch
from pyro.nn import PyroModule


class Guide(PyroModule):
    def __init__(self, model, device: Optional[Union[torch.device, int]] = None):
        """Variational Distribution base class.

        Parameters
        ----------
        model : cellij.Model
            A cellij Model instance
        device : Optional[Union[torch.device, int]], optional
            Device to run the guide on, by default None
        """
        super().__init__(name="Guide")
        self.model = model
        self.device = device

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError()
