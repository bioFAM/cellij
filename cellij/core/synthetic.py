"""Data module."""
import itertools
import logging
import math
from typing import List, Optional, Tuple

import anndata as ad
import mudata as mu
import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class DataGenerator:
    """Synthetic data generator."""

    def __init__(
        self,
        n_samples: int,
        n_features: List[int],
        likelihoods: Optional[List[str]] = None,
        n_fully_shared_factors: int = 2,
        n_partially_shared_factors: int = 15,
        n_private_factors: int = 3,
        n_covariates: int = 0,
        factor_size_dist: str = "uniform",
        factor_size_params: Optional[Tuple[float, float]] = None,
        n_active_factors: float = 1.0,
        **kwargs,
    ):
        """Generate synthetic data.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        n_features : List[int]
            Number of feature for each feature group.
        likelihoods : List[str], optional
            Likelihoods for each feature group,
            "Normal" or "Bernoulli", by default None.
        n_fully_shared_factors : int, optional
            Number of fully shared latent factors,
            by default 2.
        n_partially_shared_factors : int, optional
            Number of partially shared factors,
            by default 15.
        n_private_factors : int, optional
            Number of private factors, by default 3.
        n_covariates : int, optional
            Number of additional covariates, by default 0.
        factor_size_dist : str, optional
            Distribution of the number of active factor loadings,
            either "uniform" or "gamma",
            by default "uniform".
        factor_size_params : Tuple[float], optional
            Parameters for the distribution of the number
            of active factor loadings for the latent factors,
            by default None.
        n_active_factors : float, optional
            Number or fraction of active factors,
            by default 1.0 (all).
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_feature_groups = len(self.n_features)
        self.n_fully_shared_factors = n_fully_shared_factors
        self.n_partially_shared_factors = n_partially_shared_factors
        self.n_private_factors = n_private_factors
        self.n_covariates = n_covariates

        if factor_size_params is None:
            if factor_size_dist == "uniform":
                logger.warning(
                    "Using a uniform distribution with parameters 0.05 and 0.15 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (0.05, 0.15)
            elif factor_size_dist == "gamma":
                logger.warning(
                    "Using a uniform distribution with shape of 1 and scale of 50 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (1.0, 50.0)

        if isinstance(factor_size_params, tuple):
            factor_size_params = [
                factor_size_params for _ in range(self.n_feature_groups)
            ]

        self.factor_size_params = factor_size_params
        self.factor_size_dist = factor_size_dist

        # custom assignment
        if likelihoods is None:
            likelihoods = ["Normal" for _ in range(self.n_feature_groups)]
        self.likelihoods = likelihoods

        self.n_active_factors = n_active_factors

        # set upon data generation, see `generate`
        # covariates
        self.x = None
        # covariate coefficients
        self.betas = None
        # latent factors
        self.z = None
        # factor loadings
        self.ws = None
        self.sigmas = None
        self.ys = None

        self.w_masks = None
        self.noisy_w_masks = None
        self.active_factor_indices = None
        self.feature_group_factor_mask = None
        # set when introducing missingness
        self.presence_masks = None

    @property
    def n_factors(self) -> int:
        """Number of factors.

        Returns
        -------
        int
            Number of factors.
        """
        return (
            self.n_fully_shared_factors
            + self.n_partially_shared_factors
            + self.n_private_factors
        )

    def _attr_to_matrix(self, attr_name: str, axis: int = 1) -> ArrayLike:
        """Concatenate list of attributes into a single array.

        Parameters
        ----------
        attr_name : str
            Name of an attribute, e.g. ys for data.
        axis : int, optional
            Axis on which to concatenate arrays, by default 1.

        Returns
        -------
        ArrayLike
            Concatenated attributes as a single array.
        """
        attr = getattr(self, attr_name)
        if attr is not None:
            attr = np.concatenate(attr, axis=axis)
        return attr

    def _mask_to_nan(self):
        """Replace missing values with NaNs."""
        nan_masks = []
        for mask in self.presence_masks:
            nan_mask = np.array(mask, dtype=np.float32, copy=True)
            nan_mask[nan_mask == 0] = np.nan
            nan_masks.append(nan_mask)
        return nan_masks

    def _mask_to_bool(self):
        """Replace missing values with False."""
        bool_masks = []
        for mask in self.presence_masks:
            bool_mask = mask == 1.0
            bool_masks.append(bool_mask)
        return bool_masks

    @property
    def missing_ys(self):
        """List of data with missing values."""
        if self.ys is None:
            logger.warning("Generate data first by calling `generate`.")
            return []
        if self.presence_masks is None:
            logger.warning(
                "Introduce missing data first by calling `generate_missingness`."
            )
            return self.ys

        nan_masks = self._mask_to_nan()

        return [self.ys[m] * nan_masks[m] for m in range(self.n_feature_groups)]

    @property
    def y(self):
        """Data without any missing values."""
        return self._attr_to_matrix("ys")

    @property
    def missing_y(self):
        """Data without missing values."""
        return self._attr_to_matrix("missing_ys")

    @property
    def w(self):
        """Factor loadings."""
        return self._attr_to_matrix("ws")

    @property
    def w_mask(self):
        """Factor loadings activity mask."""
        return self._attr_to_matrix("w_masks")

    @property
    def noisy_w_mask(self):
        """Noisy factor loadings activity mask."""
        return self._attr_to_matrix("noisy_w_masks")

    def _generate_factor_mask(
        self,
        rng: Optional[Generator] = None,
        all_combs: bool = False,
        level: str = "features",
    ) -> ArrayLike:
        """Generate factor activity mask across groups.

        Parameters
        ----------
        rng : Optional[Generator], optional
            Random number generator, by default None
        all_combs : bool, optional
            Whether to generate all combinations of active factors,
            by default False
        level : str, optional
            'obs' or 'features', by default 'features'

        Returns
        -------
        ArrayLike
            A boolean array of groups times factors.
        """

        # n_groups = self.n_sample_groups
        if level == "features":
            n_groups = self.n_feature_groups

        if all_combs and n_groups == 1:
            logger.warning(
                f"Single `{level}` group dataset, "
                f"cannot generate factor combinations for a single `{level}` group."
            )
            all_combs = False
        if all_combs:
            logger.warning(
                "Generating all possible binary combinations of "
                f"{n_groups} variables."
            )
            self.n_fully_shared_factors = 1
            self.n_private_factors = n_groups
            self.n_partially_shared_factors = 2**n_groups - 2 - self.n_private_factors
            logger.warning(
                f"New factor configuration across `{level}` groups: "
                f"{self.n_fully_shared_factors} fully shared, "
                f"{self.n_partially_shared_factors} partially shared, "
                f"{self.n_private_factors} private factors."
            )

            return np.array(
                [list(i) for i in itertools.product([1, 0], repeat=n_groups)]
            )[:-1, :].T

        if rng is None:
            rng = np.random.default_rng()

        factor_mask = np.ones([n_groups, self.n_factors])

        for factor_idx in range(self.n_fully_shared_factors, self.n_factors):
            # exclude group subsets for partially shared factors
            if (
                factor_idx
                < self.n_fully_shared_factors + self.n_partially_shared_factors
            ):
                if n_groups > 2:
                    exclude_group_subset_size = rng.integers(1, n_groups - 1)
                else:
                    exclude_group_subset_size = 0

                exclude_group_subset = rng.choice(
                    n_groups, exclude_group_subset_size, replace=False
                )
            # exclude all but one group for private factors
            else:
                include_group_idx = rng.integers(n_groups)
                exclude_group_subset = [
                    i for i in range(n_groups) if i != include_group_idx
                ]

            for g in exclude_group_subset:
                factor_mask[g, factor_idx] = 0

        if self.n_private_factors >= n_groups:
            factor_mask[-n_groups:, -n_groups:] = np.eye(n_groups)

        return factor_mask

    def normalize(self, with_std: bool = False) -> None:
        """Normalize data.

        Parameters
        ----------
        with_std : bool, optional
            Whether to standardize the data, by default False
        """

        for m in range(self.n_feature_groups):
            if self.likelihoods[m] == "Normal":
                y = np.array(self.ys[m], dtype=np.float32, copy=True)
                y -= y.mean(axis=0)
                if with_std:
                    y_std = y.std(axis=0)
                    y = np.divide(y, y_std, out=np.zeros_like(y), where=y_std != 0)
                self.ys[m] = y

    def sigmoid(self, x):
        """Sigmoid transformation."""
        return 1.0 / (1 + np.exp(-x))

    def generate(
        self,
        seed: int = None,
        all_feature_group_combs: bool = False,
        overwrite: bool = False,
    ) -> Generator:
        """Generate synthetic data.

        Parameters
        ----------
        seed : int, optional
            Random seed, by default None
        overwrite : bool, optional
            Whether to overwrite existing data, by default False

        Returns
        -------
        Generator
            The numpy random generator that generated this data.
        """
        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        if self.ys is not None and not overwrite:
            logger.warning(
                "Data has already been generated, "
                "to generate new data please set `overwrite` to True."
            )
            return rng

        betas = []
        ws = []
        sigmas = []
        ys = []
        w_masks = []

        # may change the number of factors if all_feature_group_combs is True
        feature_group_factor_mask = self._generate_factor_mask(
            rng, all_combs=all_feature_group_combs, level="features"
        )

        # generate factor scores which lie in the latent space
        z = rng.standard_normal((self.n_samples, self.n_factors))

        if self.n_covariates > 0:
            x = rng.standard_normal((self.n_samples, self.n_covariates))

        n_active_factors = self.n_active_factors
        if n_active_factors <= 1.0:
            # if fraction of active factors convert to int
            n_active_factors = int(n_active_factors * self.n_factors)

        active_factor_indices = sorted(
            rng.choice(
                self.n_factors,
                size=math.ceil(n_active_factors),
                replace=False,
            )
        )

        for factor_idx in range(self.n_factors):
            if factor_idx not in active_factor_indices:
                feature_group_factor_mask[:, factor_idx] = 0.0

        for m in range(self.n_feature_groups):
            n_features = self.n_features[m]
            w_shape = (self.n_factors, n_features)
            w = rng.standard_normal(w_shape)
            w_mask = np.zeros(w_shape)

            fraction_active_features = {
                "gamma": lambda shape, scale: (
                    rng.gamma(shape, scale, self.n_factors) + 20
                )
                / n_features,
                "uniform": lambda low, high: rng.uniform(low, high, self.n_factors),
            }[self.factor_size_dist](
                self.factor_size_params[m][0], self.factor_size_params[m][1]
            )

            for factor_idx, faft in enumerate(fraction_active_features):
                if feature_group_factor_mask[m, factor_idx] > 0:
                    w_mask[factor_idx] = rng.choice(2, n_features, p=[1 - faft, faft])

            # set small values to zero
            tiny_w_threshold = 0.1
            w_mask[np.abs(w) < tiny_w_threshold] = 0.0
            w = w_mask * w
            # add some noise to avoid exactly zero values
            w = np.where(
                np.abs(w) < tiny_w_threshold, w + rng.standard_normal(w_shape) / 100, w
            )
            assert ((np.abs(w) > tiny_w_threshold) * 1.0 == w_mask).all()

            y_loc = np.matmul(z, w)

            if self.n_covariates > 0:
                beta_shape = (self.n_covariates, n_features)
                # reduce effect of betas by scaling them down
                beta = rng.standard_normal(beta_shape) / 10
                y_loc = y_loc + np.matmul(x, beta)
                betas.append(beta)

            # generate feature sigmas
            sigma = 1.0 / np.sqrt(rng.gamma(10.0, 1.0, n_features))

            if self.likelihoods[m] == "Normal":
                y = rng.normal(loc=y_loc, scale=sigma)
            else:
                y = rng.binomial(1, self.sigmoid(y_loc))

            ws.append(w)
            sigmas.append(sigma)
            ys.append(y)
            w_masks.append(w_mask)

        if self.n_covariates > 0:
            self.x = x
            self.betas = betas
        self.z = z
        self.ws = ws
        self.w_masks = w_masks
        self.sigmas = sigmas
        self.ys = ys
        self.active_factor_indices = active_factor_indices
        self.feature_group_factor_mask = feature_group_factor_mask

        return rng

    def get_noisy_mask(
        self,
        rng: Optional[Generator] = None,
        noise_fraction: float = 0.1,
        feature_group_indices: Optional[List[int]] = None,
    ) -> List[ArrayLike]:
        """Generate noisy factor loadings mask.

        Parameters
        ----------
        rng : Optional[Generator], optional
            Random number generator, by default None
        noise_fraction : float, optional
            Fraction of noise, by default 0.1
        feature_group_indices : Optional[List[int]], optional
            Feature groups on which to introduce noise,
            by default None (all)

        Returns
        -------
        List[ArrayLike]
            A list of boolean arrays.
        """
        if rng is None:
            rng = np.random.default_rng()

        if feature_group_indices is None:
            logger.warning(
                "Parameter `feature_group_indices` set to None, "
                "adding noise to all views."
            )
            feature_group_indices = list(range(self.n_feature_groups))

        noisy_w_masks = [np.array(mask, copy=True) for mask in self.w_masks]

        if len(feature_group_indices) == 0:
            logger.warning(
                "Parameter `feature_group_indices` "
                "set to an empty list, removing information from all views."
            )
            self.noisy_w_masks = [np.ones_like(mask) for mask in noisy_w_masks]
            return self.noisy_w_masks

        for m in range(self.n_feature_groups):
            noisy_w_mask = noisy_w_masks[m]

            if m in feature_group_indices:
                fraction_active_cells = (
                    noisy_w_mask.mean(axis=1).sum()
                    / self.feature_group_factor_mask[0].sum()
                )
                for factor_idx in range(self.n_factors):
                    active_cell_indices = noisy_w_mask[factor_idx, :].nonzero()[0]
                    # if all features turned off
                    # => simulate random noise in terms of false positives only
                    if len(active_cell_indices) == 0:
                        logger.warning(
                            f"Factor {factor_idx} is completely off, introducing "
                            f"{(100 * fraction_active_cells):.2f} false positives."
                        )
                        active_cell_indices = rng.choice(
                            self.n_features[m],
                            int(self.n_features[m] * fraction_active_cells),
                            replace=False,
                        )

                    inactive_cell_indices = (
                        noisy_w_mask[factor_idx, :] == 0
                    ).nonzero()[0]
                    n_noisy_cells = int(noise_fraction * len(active_cell_indices))
                    swapped_indices = zip(
                        rng.choice(
                            len(active_cell_indices), n_noisy_cells, replace=False
                        ),
                        rng.choice(
                            len(inactive_cell_indices), n_noisy_cells, replace=False
                        ),
                    )

                    for on_idx, off_idx in swapped_indices:
                        noisy_w_mask[factor_idx, active_cell_indices[on_idx]] = 0.0
                        noisy_w_mask[factor_idx, inactive_cell_indices[off_idx]] = 1.0

            else:
                noisy_w_mask.fill(0.0)

        self.noisy_w_masks = noisy_w_masks
        return self.noisy_w_masks

    def generate_missingness(
        self,
        random_fraction: float = 0.0,
        n_incomplete_samples: int = 0,
        n_incomplete_features: int = 0,
        missing_fraction_incomplete_features: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Generate missingness pattern.

        Parameters
        ----------
        random_fraction : float, optional
            Fraction of missing data at random, by default 0.0
        n_incomplete_samples : int, optional
            Number of incomplete samples, by default 0
        n_incomplete_features : int, optional
            Number of incomplete features, by default 0
        missing_fraction_incomplete_features : float, optional
            Missingness fraction of incomplete features, by default 0.0
        seed : int, optional
            Random seed, by default None

        Returns
        -------
        Generator
            The numpy random generator that generated this data.
        """

        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        n_incomplete_samples = int(n_incomplete_samples)
        n_incomplete_features = int(n_incomplete_features)

        sample_view_mask = np.ones((self.n_samples, self.n_feature_groups))
        missing_sample_indices = rng.choice(
            self.n_samples, n_incomplete_samples, replace=False
        )

        # partially missing samples
        for ms_idx in missing_sample_indices:
            if self.n_feature_groups > 1:
                exclude_view_subset_size = rng.integers(1, self.n_feature_groups)
            else:
                exclude_view_subset_size = 0
            exclude_view_subset = rng.choice(
                self.n_feature_groups, exclude_view_subset_size, replace=False
            )
            sample_view_mask[ms_idx, exclude_view_subset] = 0

        mask = np.repeat(sample_view_mask, self.n_features, axis=1)

        # partially missing features
        missing_feature_indices = rng.choice(
            sum(self.n_features), n_incomplete_features, replace=False
        )

        for mf_idx in missing_feature_indices:
            random_sample_indices = rng.choice(
                self.n_samples,
                int(self.n_samples * missing_fraction_incomplete_features),
                replace=False,
            )
            mask[random_sample_indices, mf_idx] = 0

        # remove random fraction
        mask *= rng.choice([0, 1], mask.shape, p=[random_fraction, 1 - random_fraction])

        view_feature_offsets = [0] + np.cumsum(self.n_features).tolist()
        masks = []
        for offset_idx in range(len(view_feature_offsets) - 1):
            start_offset = view_feature_offsets[offset_idx]
            end_offset = view_feature_offsets[offset_idx + 1]
            masks.append(mask[:, start_offset:end_offset])

        self.presence_masks = masks

        return rng

    def to_mdata(self) -> mu.MuData:
        feature_group_names = []
        ad_dict = {}
        for m in range(self.n_feature_groups):
            adata = ad.AnnData(
                self.ys[m],
                dtype=np.float32,
            )
            adata.var_names = f"feature_group_{m}:" + adata.var_names
            adata.varm["w"] = self.ws[m].T
            adata.varm["w_mask"] = self.w_masks[m].T
            feature_group_name = f"feature_group_{m}"
            ad_dict[feature_group_name] = adata
            feature_group_names.append(feature_group_name)

        mdata = mu.MuData(ad_dict)
        mdata.uns["likelihoods"] = dict(zip(feature_group_names, self.likelihoods))
        mdata.uns["n_active_factors"] = self.n_active_factors

        mdata.obsm["z"] = self.z

        return mdata


if __name__ == "__main__":
    dg = DataGenerator(
        n_samples=200,
        n_features=[400, 400],
        likelihoods="Normal",
    )
    dg.generate()
