import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroModule

'''
TODO: Every model class must inherit from Abstract Meta Model Class, e.g.

class GenerativeModel(PyroModule):
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
        super().__init__(name="GenerativeModel")
        self.device = device

    def get_plates(self, *args, **kwargs) -> Dict[str, pyro.plate]:
        raise NotImplementedError()
    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError()
'''


class MOFA_Model(PyroModule):
    def __init__(self, n_factors: int):
        super().__init__(name="MOFA_Model")
        self.n_factors = n_factors

    def forward(self, X):
        """Generative model for MOFA."""
        plates = self.get_plates()

        with plates["obs"]:
            z = pyro.sample("z", dist.Normal(torch.zeros(self.n_factors), torch.ones(self.n_factors)))

        with plates["factors"]:
            with plates["features"]:
                w = pyro.sample("w", dist.Normal(torch.zeros(1), torch.ones(1)))

        with plates["features"]:
            sigma = pyro.sample("sigma", dist.InverseGamma(torch.tensor(3.0), torch.tensor(1.0)))

        with plates["obs"]:
            pyro.sample(
                "data",
                dist.Normal(z @ w, torch.sqrt(sigma)),
                obs=X.T,
            )

    def get_plates(self):
        return {
            "obs": pyro.plate("obs", 136, dim=-2),
            "factors": pyro.plate("factors", self.n_factors, dim=-2),
            "features": pyro.plate("features", 5000, dim=-1),
        }
