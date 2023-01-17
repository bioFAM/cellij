from __future__ import annotations

import mfmf
import pyro
import torch

# import logging
# import sys
# from collections.abc import Iterable
# from importlib.metadata import version
# from typing import List, Optional, Union
# import itertools
# import anndata
# import matplotlib.pyplot as plt
# import muon
# import numpy as np
# import pandas as pd
# from pyro.infer import SVI
# import os


class MOFA(mfmf.core.FactorModel):
    def __init__(
        self,
        n_factors: int,
        optimizer: pyro.optim.optim.PyroOptim,
        loss: pyro.infer.ELBO,
        guide: str = "AutoNormal",
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        super().__init__(
            n_factors=n_factors,
            optimizer=optimizer,
            loss=loss,
            guide=guide,
            dtype=dtype,
            device=device,
        )
