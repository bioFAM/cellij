from __future__ import annotations


class Covariates:
    def __init__(self, flavour: str, covariates: dict) -> None:

        if not isinstance(flavour):
            raise TypeError("Parameter 'flavour' must be a string.")

        if not isinstance(covariates, dict):
            raise TypeError("Parameter 'covariates' must be a dictionary.")

        if len(covariates.keys()) < 2:
            raise ValueError("The given covariates must form at least two groups.")

        self.flavour = flavour
        self.covariates = covariates


class UnorderedCovariates(Covariates):
    def __init__(self, covariates: List[str]):
        self.covariates = covariates
