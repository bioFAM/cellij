from __future__ import annotations

import mfmf
import numpy as np
import pandas as pd
import sys
import warnings

# from itertools import Iterable
from typing import Optional, Union, List, Iterable

###############################################################################
# from mofax.utils ############################################################


def padjust_fdr(xs):
    """
    Adjust p-values using the BH procedure
    """
    from scipy.stats import rankdata

    ranked_p_values = rankdata(xs)
    fdr = xs * len(xs) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


def calculate_r2(Z, W, Y):

    # print(np.sum(np.isnan(Z)))
    # print(np.sum(np.isnan(W)))

    # print("w", W.shape)
    # print(f"{W.shape=}")
    # print(f"{Z.shape=}")

    a = np.nansum((Y - Z.T.dot(W)) ** 2.0)
    b = np.nansum(Y**2)
    r2 = (1.0 - a / b) * 100
    # print(a, b, r2)

    return r2


def maybe_factor_indices_to_factors(x):
    if x is None:
        return None
    elif isinstance(x, int):
        return f"Factor{x+1}"
    elif isinstance(x, str):
        return x
    return [f"Factor{i+1}" if isinstance(i, int) else i for i in x]


###############################################################################
# from mofax.core.mofa_mode() #################################################


def get_shape(model, groups=None, views=None):
    """
    Get the shape of all the data, samples (cells) and features pulled across groups and views.
    Parameters
    ----------
    groups : optional
        List of groups to consider
    views : optional
        List of views to consider
    """
    groups = model._check_groups(groups)
    views = model._check_views(views)

    sum_features = sum(
        [
            len(model.views[view_name].feature_names)
            for view_name in model.view_names
            if view_name in views
        ]
    )

    sum_obs = sum(
        len(og["observations"]) for og in model.obs_groups if og["group_name"] in groups
    )

    shape = (sum_obs, sum_features)

    return shape


def get_samples(model, groups=None):
    """
    Get the sample metadata table (sample ID and its respective group)
    Parameters
    ----------
    groups : optional
        List of groups to consider
    """

    samples = pd.DataFrame()

    for og in model.obs_groups:

        if groups is None or og["group_name"] in groups:

            samples = pd.concat(
                [
                    samples,
                    pd.DataFrame(
                        {
                            "group": [og["group_name"]] * len(og["observations"]),
                            "sample": og["observations"],
                        }
                    ),
                ],
                axis=0,
            )

    samples = samples.sort_values("sample")

    return samples


# Alias samples as cells
def get_cells(model, groups=None):
    """
    Get the cell metadata table (cell ID and its respective group)
    Parameters
    ----------
    groups : optional
        List of groups to consider
    """
    cells = model.get_samples(groups)
    cells.columns = ["group", "cell"]
    return cells


def get_features(model, views=None):
    """
    Get the features metadata table (feature name and its respective view)
    Parameters
    ----------
    views : optional
        List of views to consider
    """
    views = model._check_views(views)

    features = pd.DataFrame()

    for fg in model.feature_groups:

        if views is None or fg["associated_view"] in views:

            features = pd.concat(
                [
                    features,
                    pd.DataFrame(
                        {
                            "view": [fg["associated_view"]] * len(fg["features"]),
                            "feature": fg["features"],
                        }
                    ),
                ],
                axis=0,
            )

    return features


def get_groups(model):
    """
    Get the groups names
    """
    return model.groups


def get_views(model):
    """
    Get the views names
    """
    return model.view_names


def get_top_features(
    model,
    factors: Union[int, List[int]] = None,
    views: Union[str, int, List[str], List[int]] = None,
    n_features: int = None,
    clip_threshold: float = None,
    scale: bool = False,
    absolute_values: bool = False,
    only_positive: bool = False,
    only_negative: bool = False,
    per_view: bool = True,
    df: bool = False,
):
    """
    Fetch a list of top feature names
    Parameters
    ----------
    factors : optional
        Factors to use (all factors in the model by default)
    view : options
        The view to get the factor weights for (first view by default)
    n_features : optional
        Number of features for each factor by their absolute value (10 by default)
    clip_threshold : optional
        Absolute weight threshold to clip all values to (no threshold by default)
    absolute_values : optional
        If to fetch absolute weight values
    only_positive : optional
        If to fetch only positive weights
    only_negative : optional
        If to fetch only negative weights
    per_view : optional
        Get n_features per view rather than globally (True by default)
    df : optional
        Boolean value if to return a DataFrame
    """
    views = model._check_views(views)
    factor_indices, factors = model._check_factors(factors, unique=True)
    n_features_default = 10

    # Fetch weights for the relevant factors
    w = (
        mfmf.mofax_support.get_weights(
            model=model,
            views=views,
            factors=factors,
            df=True,
            absolute_values=absolute_values,
            concatenate_views=True,
        )
        .rename_axis("feature")
        .reset_index()
    )
    wm = w.melt(id_vars="feature", var_name="factor", value_name="value")
    wm = wm.assign(value_abs=lambda x: x.value.abs())
    wm["factor"] = wm["factor"].astype("category")
    wm = (
        wm.set_index("feature")
        .join(model.features_metadata.loc[:, ["view"]], how="left")
        .reset_index()
    )

    if only_positive and only_negative:
        print("Please specify either only_positive or only_negative")
        sys.exit(1)
    elif only_positive:
        wm = wm[wm.value > 0]
    elif only_negative:
        wm = wm[wm.value < 0]

    if n_features is None and clip_threshold is not None:
        wm = wm[wm.value_abs >= clip_threshold]
    else:
        if n_features is None:
            n_features = n_features_default
        # Get a subset of features
        if per_view:
            wm = wm.sort_values(["factor", "value_abs"], ascending=False).groupby(
                ["factor", "view"]
            )
        else:
            wm = wm.sort_values(["factor", "value_abs"], ascending=False).groupby(
                ["factor", "view"]
            )
        # Use clip threshold if provided
        if clip_threshold is None:
            wm = wm.head(n_features).reset_index()
        else:
            wm = wm[wm.value_abs >= clip_threshold].head(n_features)

    if df:
        return wm

    features = wm.feature.unique()
    return features


def get_factors(
    model,
    groups: Union[str, int, List[str], List[int]] = None,
    factors: Optional[Union[int, List[int], str, List[str]]] = None,
    df: bool = False,
    concatenate_groups: bool = True,
    scale: bool = False,
    absolute_values: bool = False,
):
    """
    Get the matrix with factors as a NumPy array or as a DataFrame (df=True).
    Parameters
    ----------
    groups : optional
        List of groups to consider
    factors : optional
        Indices of factors to consider
    df : optional
        Boolean value if to return the factor matrix Z as a (wide) pd.DataFrame
    concatenate_groups : optional
        If concatenate Z matrices (True by default)
    scale : optional
        If return values scaled to zero mean and unit variance
        (per factor when concatenated or per factor and per group otherwise)
    absolute_values : optional
        If return absolute values for weights
    """

    groups = model._check_groups(groups)
    factor_indices, factors = model._check_factors(factors)

    z = {}
    for og in model.obs_groups:

        group_name = og["group_name"]

        z[group_name] = model.results[group_name]["z"][:, factor_indices]

        if not concatenate_groups:
            if scale:
                z[group_name] = (z[group_name] - z[group_name].mean(axis=0)) / z[
                    group_name
                ].std(axis=0)
            if absolute_values:
                z[group_name] = np.absolute(z[group_name])

        if df:
            z[group_name] = pd.DataFrame(z[group_name])
            z[group_name].columns = factors
            z[group_name].index = og["observations"]

    # concatenate views if requested
    if concatenate_groups:

        if not df:
            z = np.concatenate([x for x in z.values()], axis=0)
        elif df:
            z = pd.concat([x for x in z.values()], axis=0)

        if scale:
            z = (z - z.mean(axis=0)) / z.std(axis=0)
        if absolute_values:
            z = np.absolute(z)

    return z


def get_weights(
    model,
    views: Union[str, int, List[str], List[int]] = None,
    factors: Union[int, List[int]] = None,
    df: bool = False,
    scale: bool = False,
    concatenate_views: bool = True,
    absolute_values: bool = False,
):
    """
    Fetch the weight matrices
    Parameters
    ----------
    views : optional
        List of views to consider
    factors : optional
        Indices of factors to use
    df : optional
        Boolean value if to return W matrix as a (wide) pd.DataFrame
    scale : optional
        If return values scaled to zero mean and unit variance
        (per factor when concatenated or per factor and per view otherwise)
    concatenate_weights : optional
        If concatenate W matrices (True by default)
    absolute_values : optional
        If return absolute values for weights
    """

    if not model.is_trained:

        raise NotImplementedError("Must be a trained model.")

    views = model._check_views(views)
    factor_indices, factors = model._check_factors(factors, unique=True)

    w = {}
    for view_name in views:

        w[view_name] = model.results[view_name]["w"].T[:, factor_indices]

        if not concatenate_views:
            if scale:
                w[view_name] = (w[view_name] - w[view_name].mean(axis=0)) / w[
                    view_name
                ].std(axis=0)
            if absolute_values:
                w[view_name] = np.absolute(w[view_name])

        if df:
            w[view_name] = pd.DataFrame(w[view_name])
            w[view_name].columns = factors
            w[view_name].index = model.views[view_name].feature_names

    # concatenate views if requested
    if concatenate_views:

        if not df:
            w = np.concatenate([x for x in w.values()], axis=0)
        elif df:
            w = pd.concat([x for x in w.values()], axis=0)

        if scale:
            w = (w - w.mean(axis=0)) / w.std(axis=0)
        if absolute_values:
            w = np.absolute(w)

    return w


def get_data(
    model,
    views: Optional[Union[str, int]] = None,
    features: Optional[Union[str, List[str]]] = None,
    groups: Optional[Union[str, int, List[str], List[int]]] = None,
    df: bool = False,
):
    """
    Fetch the training data
    Parameters
    ----------
    view : optional
        view to consider
    features : optional
        Features to consider (from one view)
    groups : optional
        groups to consider
    df : optional
        Boolean value if to return Y matrix as a DataFrame
    """

    # Sanity checks
    groups = model._check_groups(groups)
    views = model._check_views(views)

    # If features is None (default), return all by default
    pd_features = model.get_features(views)
    if features is None:
        features = pd_features.feature.values

    # If a sole feature name is used, wrap it in a list
    if not isinstance(features, Iterable) or isinstance(features, str):
        features = [features]
    else:
        features = list(set(features))  # make feature names unique

    f_i = np.where(pd_features.feature.isin(features))[0]
    assert len(f_i) > 0, "Requested features are not found"
    pd_features = pd_features.loc[f_i]

    # Create numpy array
    # y = [model.data[view][g][:, :] for g in groups]
    ym = []
    for m in views:
        ym.append(np.concatenate([model.data[m][g][:, f_i] for g in groups], axis=0))
    y = np.concatenate(ym, axis=1)

    # Convert output to pandas data.frame
    if df:
        y = pd.DataFrame(y)
        y.columns = pd_features.feature.values
        y.index = np.concatenate(tuple(model.samples[g] for g in groups))

    return y


def run_umap(
    model,
    groups: Union[str, int, List[str], List[int]] = None,
    factors: Union[int, List[int]] = None,
    n_neighbors: int = 10,
    min_dist: float = 0.5,
    spread: float = 1.0,
    random_state: int = 42,
    **kwargs,
) -> None:
    """
    Run UMAP on the factor space
    Parameters
    ----------
    n_neighbors : optional
        UMAP parameter: number of neighbors.
    min_dist
        UMAP parameter: the effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out.
    spread
        UMAP parameter: the effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    random_state
        random seed
    """
    import umap

    # Get factors
    data = model.get_factors(groups, factors, df=True)

    embedding = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        random_state=random_state,
        **kwargs,
    ).fit_transform(data)

    # create pandas dataframe
    pd_umap = pd.DataFrame(embedding)
    pd_umap.columns = ["UMAP" + str(i + 1) for i in range(embedding.shape[1])]
    pd_umap.index = data.index

    return pd_umap


def fetch_values(model, variables: Union[str, List[str]], unique: bool = True):
    """
    Fetch metadata column, factors, or feature values
    as well as covariates.
    Shorthand to get_data, get_factors, metadata, and covariates calls.
    Parameters
    ----------
    variables : str
        Features, metadata columns, or factors (FactorN) to fetch.
        For MEFISTO models with covariates, covariates are accepted
        such as 'covariate1' or 'covariate1_transformed'.
    """
    # If a sole variable name is used, wrap it in a list
    if not isinstance(variables, Iterable) or isinstance(variables, str):
        variables = [variables]

    # Remove None values and duplicates
    variables = [i for i in variables if i is not None]
    # Transform integers to factors
    variables = maybe_factor_indices_to_factors(variables)
    if unique:
        variables = pd.Series(variables).drop_duplicates().tolist()

    var_meta = list()
    var_features = list()
    var_factors = list()
    var_covariates = list()

    # Split all the variables into metadata and features
    for i, var in enumerate(variables):
        if var in model.metadata.columns:
            var_meta.append(var)
        elif var.capitalize().startswith("Factor"):
            # Unify factor naming
            variables[i] = var.capitalize()
            var_factors.append(var.capitalize())
        elif (
            model.covariates_names is not None
            and (
                var in model.covariates_names
                or var in [f"{cov}_transformed" for cov in model.covariates_names]
            )
            and model.covariates is not None
        ):
            var_covariates.append(var)
        else:
            var_features.append(var)

    var_list = list()
    if len(var_meta) > 0:
        var_list.append(model.metadata[var_meta])
    if len(var_features) > 0:
        var_list.append(model.get_data(features=var_features, df=True))
    if len(var_factors) > 0:
        var_list.append(model.get_factors(factors=var_factors, df=True))
    if len(var_covariates) > 0:
        var_list.append(model.covariates[var_covariates])

    # Return a DataFrame with columns ordered as requested
    return pd.concat(var_list, axis=1).loc[:, variables]


def _check_views(model, views):
    if views is None:
        views = model.view_names
    # single view provided as a string
    elif isinstance(views, str):
        views = [views]

    # single view provided as an integer
    elif isinstance(views, int):
        views = [model.view_names[views]]

    # multiple views provided as an iterable
    elif isinstance(views, Iterable) and not isinstance(views, str):

        # (to-do) check that all elements are of the same type

        # iterable of booleans
        if all([isinstance(m, bool) for m in views]):
            raise ValueError(
                f"Please provide view names as string or view indices as integers, boolean values are not accepted. Group names of this model are {', '.join(model.view_names)}."
            )
        # iterable of integers
        elif all([isinstance(m, int) for m in views]):
            views = [model.view_names[m] if isinstance(m, int) else m for m in views]
        # iterable of strings
        elif all([isinstance(m, str) for m in views]):
            assert set(views).issubset(
                set(model.view_names)
            ), f"some of the elements of the 'views' are not valid views. Views names of this model are {', '.join(model.view_names)}."
        else:
            raise ValueError(
                "elements of the 'view' vector have to be either integers or strings"
            )
    else:
        raise ValueError("views argument not recognised")

    return views


def _check_groups(model, groups):
    if groups is None:
        groups = model.groups
    # single group provided as a string
    elif isinstance(groups, str):
        groups = [groups]

    # single group provided as an integer
    elif isinstance(groups, int):
        groups = [model.groups[groups]]

    # multiple groups provided as an iterable
    elif isinstance(groups, Iterable) and not isinstance(groups, str):

        # (to-do) check that all elements are of the same type

        # iterable of booleans
        if all([isinstance(g, bool) for g in groups]):
            raise ValueError(
                f"Please provide group names as string or group indices as integers, boolean values are not accepted. Group names of this model are {', '.join(model.groups)}."
            )
        # iterable of integers
        elif all([isinstance(g, int) for g in groups]):
            groups = [model.groups[g] if isinstance(g, int) else g for g in groups]
        # iterable of strings
        elif all([isinstance(g, str) for g in groups]):
            assert set(groups).issubset(
                set(model.groups)
            ), f"some of the elements of the 'groups' are not valid groups. Group names of this model are {', '.join(model.groups)}."
        else:
            raise ValueError(
                "elements of the 'group' vector have to be either integers or strings"
            )
    else:
        raise ValueError("groups argument not recognised")

    return groups


# def _check_grouping(model, groups, grouping_instance):
#     assert grouping_instance in ["groups", "views"]
#     # Use all groups if no specific groups are requested
#     if groups is None:
#         if grouping_instance == "groups":
#             groups = model.groups
#         elif grouping_instance == "views":
#             groups = model.view_names
#     # If a sole group name is used, wrap it in a list
#     if not isinstance(groups, Iterable) or isinstance(groups, str):
#         groups = [groups]
#     # Do not accept boolean values
#     if any([isinstance(g, bool) for g in groups]):
#         if grouping_instance == "groups":
#             raise ValueError(
#                 f"Please provide relevant group names. Boolean values are not accepted. Group names of this model are {', '.join(model.groups)}."
#             )
#         elif grouping_instance == "views":
#             raise ValueError(
#                 f"Please provide relevant view names. Boolean values are not accepted. View names of this model are {', '.join(model.view_names)}."
#             )
#     # Convert integers to group names
#     if grouping_instance == "groups":
#         groups = [model.groups[g] if isinstance(g, int) else g for g in groups]
#     elif grouping_instance == "views":
#         groups = [model.view_names[g] if isinstance(g, int) else g for g in groups]
#     return groups


def _check_factors(model, factors, unique=False):
    # Use all factors by default
    if factors is None:
        factors = list(range(model.n_factors))
    # If one factor is used, wrap it in a list
    if not isinstance(factors, Iterable) or isinstance(factors, str):
        factors = [factors]
    if unique:
        factors = list(set(factors))
    # Convert factor names (FactorN) to factor indices (N-1)
    factor_indices = [
        int(fi.replace("Factor", "")) - 1 if isinstance(fi, str) else fi
        for fi in factors
    ]
    factors = [f"Factor{fi+1}" if isinstance(fi, int) else fi for fi in factors]

    return (factor_indices, factors)


# Variance explained (R2)


def calculate_variance_explained(
    model,
    # factor_index: int,
    factors: Optional[Union[int, List[int], str, List[str]]] = None,
    groups: Optional[Union[str, int, List[str], List[int]]] = None,
    views: Optional[Union[str, int, List[str], List[int]]] = None,
    group_label: Optional[str] = None,
    per_factor: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Calculate the variance explained estimates for each factor in each view and/or group.
    Allow also for predefined groups
    Parameters
    ----------
    factors : optional
        List of factors to consider (default is None, all factors)
    groups : optional
        List of groups to consider (default is None, all groups)
    views : optional
        List of views to consider (default is None, all views)
    group_label : optional
        Group label to split samples by (default is None)
    per_factor : optional
        If calculate R2 per factor or for all factors (default)
    """

    if not model.is_trained:
        raise ValueError("Model is not trained yet.")

    groups = model._check_groups(groups)
    views = model._check_views(views)
    factor_indices, factor_names = model._check_factors(factors)

    if per_factor is None:
        per_factor = False

    r2_df = pd.DataFrame()

    # use model groups
    if group_label is None or group_label == "group":
        for view in views:
            for group in groups:
                if per_factor:
                    for f_ind_name in zip(factor_indices, factor_names):
                        factor_index, factor_name = f_ind_name
                        r2 = calculate_r2(
                            Z=np.array(model.results[group]["z"].T[[factor_index], :]),
                            W=np.array(model.results[view]["w"][[factor_index], :]),
                            Y=np.array(model.results[view][group]["obs"]),
                        )
                        r2_df = pd.concat(
                            [
                                r2_df,
                                pd.DataFrame(
                                    {
                                        "View": view,
                                        "Group": group,
                                        "R2": r2,
                                        "Factor": factor_name,
                                    }
                                ),
                            ],
                            ignore_index=True,
                        )
                else:
                    r2 = calculate_r2(
                        Z=np.array(model.results[group]["z"].T[factor_indices, :]),
                        W=np.array(model.results[view]["w"][factor_indices, :]),
                        Y=np.array(model.results[view][group]["obs"]),
                    )
                    r2_df = pd.concat(
                        [
                            r2_df,
                            pd.DataFrame(
                                {"View": [view], "Group": [group], "R2": [r2]}
                            ),
                        ],
                        ignore_index=True,
                    )

    # use custom groups
    # note that when calculating for a custom set of groups,
    # the Factor matrix (Z) has to be merged and then split according to the new grouping of samples
    else:
        custom_groups = model.samples_metadata[group_label].unique()
        samples_groups = model.samples_metadata[group_label]

        z = np.concatenate(
            [model.results[group]["z"][:, :] for group in groups], axis=1
        )

        z_custom = dict()
        for group in custom_groups:
            z_custom[group] = z[:, np.where(samples_groups == group)[0]]
        del z

        for view in views:

            y_view = np.concatenate(
                [model.results[view][group]["obs"][:, :] for group in groups], axis=0
            )

            data_view = dict()
            for group in custom_groups:
                data_view[group] = y_view[np.where(samples_groups == group)[0], :]

            for group in custom_groups:
                if per_factor:
                    for f_ind_name in zip(factor_indices, factor_names):
                        factor_index, factor_name = f_ind_name
                        r2 = calculate_r2(
                            Z=np.array(z_custom[group][[factor_index], :]),
                            W=np.array(model.results[view]["w"][[factor_index], :]),
                            Y=np.array(data_view[group]),
                        )
                        r2_df = pd.concat(
                            [
                                r2_df,
                                pd.DataFrame(
                                    {
                                        "View": view,
                                        "Group": group,
                                        "R2": r2,
                                        "Factor": factor_name,
                                    }
                                ),
                            ],
                            ignore_index=True,
                        )
                else:
                    r2 = calculate_r2(
                        Z=np.array(z_custom[group][factor_indices, :]),
                        W=np.array(model.results[view]["w"][factor_indices, :]),
                        Y=np.array(data_view[group]),
                    )
                    r2_df = pd.concat(
                        [r2_df, pd.DataFrame({"View": view, "Group": group, "R2": r2})],
                        ignore_index=True,
                    )
    return r2_df


def get_variance_explained(
    model,
    factors: Optional[Union[int, List[int], str, List[str]]] = None,
    groups: Optional[Union[str, int, List[str], List[int]]] = None,
    views: Optional[Union[str, int, List[str], List[int]]] = None,
) -> pd.DataFrame:
    """
    Get variance explained estimates (R2) for each factor across  view(s) and/or group(s).
    factors : optional
        List of factors to consider (all by default)
    groups : optional
        List of groups to consider (all by default)
    views : optional
        List of views to consider (all by default)
    """

    r2 = pd.DataFrame()
    factor_indices, _ = model._check_factors(factors)
    for k in factor_indices:
        tmp = model.calculate_variance_explained(factors=k, groups=groups, views=views)
        tmp["Factor"] = "Factor" + str(k + 1)
        r2 = pd.concat([r2, tmp])

    # Subset
    if factors is not None:
        _, factors = model._check_factors(factors)
        r2 = r2[r2.Factor.isin(factors)]
    if groups is not None:
        groups = model._check_groups(groups)
        r2 = r2[r2.Group.isin(groups)]
    if views is not None:
        view = model._check_views(views)
        r2 = r2[r2.View.isin(views)]

    return r2


def _get_factor_r2_null(
    model,
    factor_index: int,
    groups_df: Optional[pd.DataFrame],
    group_label: Optional[str],
    n_iter=100,
    return_full=False,
    return_true=False,
    return_pvalues=True,
    fdr=True,
) -> pd.DataFrame:

    r2_df = pd.DataFrame()

    if groups_df is None and group_label is None:
        group_label = "group"

    if groups_df is None:
        groups_df = model.samples_metadata.loc[:, [group_label]]

    custom_groups = groups_df.iloc[:, 0].unique()

    z = np.concatenate(
        [model.results[og["group_name"]]["z"][:, :] for og in model.obs_groups], axis=1
    )

    for i in range(n_iter + 1):
        # Calculate true group assignment for iteration 0
        if i > 0:
            groups_df.iloc[:, 0] = groups_df.iloc[:, 0].sample(frac=1).values

        z_custom = dict()
        for group in custom_groups:
            z_custom[group] = z[:, np.where(groups_df.iloc[:, 0] == group)[0]]

        for view in model.view_names:

            y_view = np.concatenate(
                [model.results[view][group]["obs"][:, :] for group in model.groups],
                axis=0,
            )

            data_view = dict()
            for group in custom_groups:
                data_view[group] = y_view[np.where(groups_df.iloc[:, 0] == group)[0], :]

            for group in custom_groups:
                crossprod = np.array(z_custom[group][[factor_index], :]).T.dot(
                    np.array(model.expectations["W"][view][[factor_index], :])
                )
                y = np.array(data_view[group])
                a = np.sum((y - crossprod) ** 2)
                b = np.sum(y**2)
                r2_df = pd.concat(
                    [
                        r2_df,
                        pd.DataFrame(
                            {
                                "View": view,
                                "Group": group,
                                "Factor": f"Factor{factor_index+1}",
                                "R2": 1 - a / b,
                                "Iteration": i,
                            }
                        ),
                    ],
                    ignore_index=True,
                )

    if return_full:
        if return_true:
            return r2_df
        else:
            return r2_df[r2_df.Iteration != 0].reset_index(drop=True)

    r2_obs = r2_df[r2_df.Iteration == 0]
    r2_df = r2_df[r2_df.Iteration != 0]

    if not return_pvalues:
        r2_null = r2_df.groupby(["Factor", "Group", "View"]).agg(
            {"R2": ["mean", "std"]}
        )
        return r2_null.reset_index()

    r2_pvalues = pd.DataFrame(
        r2_obs.set_index(["Group", "View", "Factor"])
        .loc[:, ["R2"]]
        .join(r2_df.set_index(["Group", "View", "Factor"]), rsuffix="_null")
        .groupby(["Group", "View", "Factor"])
        .apply(lambda x: np.mean(x["R2"] <= x["R2_null"]))
    )
    r2_pvalues.columns = ["PValue"]

    if fdr:
        r2_pvalues["FDR"] = padjust_fdr(r2_pvalues.PValue)
        return r2_pvalues.reset_index().sort_values("FDR", ascending=True)
    else:
        return r2_pvalues.reset_index().sort_values("PValue", ascending=True)


def _get_r2_null(
    model,
    factors: Union[int, List[int], str, List[str]] = None,
    n_iter: int = 100,
    groups_df: Optional[pd.DataFrame] = None,
    group_label: Optional[str] = None,
    return_full=False,
    return_pvalues=True,
    fdr=True,
) -> pd.DataFrame:
    factor_indices, factors = model._check_factors(factors)
    r2 = pd.DataFrame()
    for fi in factor_indices:
        r2 = pd.concat(
            [
                r2,
                pd.DataFrame(
                    model._get_factor_r2_null(
                        fi,
                        groups_df=groups_df,
                        group_label=group_label,
                        n_iter=n_iter,
                        return_full=return_full,
                        return_pvalues=return_pvalues,
                        fdr=fdr,
                    )
                ),
            ]
        )
    return r2


def get_sample_r2(
    model,
    factors: Optional[Union[str, int, List[str], List[int]]] = None,
    groups: Optional[Union[str, int, List[str], List[int]]] = None,
    views: Optional[Union[str, int, List[str], List[int]]] = None,
    df: bool = True,
) -> pd.DataFrame:
    factor_indices, factors = model._check_factors(factors, unique=True)
    groups = model._check_groups(groups)
    views = model._check_views(views)

    r2s = []
    for view in views:

        for og in model.obs_groups:

            crossprod = (
                model.results[og["group_name"]]["z"]
                .T[factor_indices, :]
                .T.dot(model.results[view]["w"][factor_indices, :])
            )

            y = np.array(model.results[view][og["group_name"]]["obs"])
            a = np.nansum((y - crossprod) ** 2.0, axis=1)
            b = np.nansum(y**2, axis=1)

            r2_df_mg = pd.DataFrame(
                {
                    "Sample": og["observations"],
                    "Group": og["group_name"],
                    "View": view,
                    "R2": (1.0 - a / b),
                }
            )

            r2s.append(r2_df_mg)

    r2_df = pd.concat(r2s, axis=0, ignore_index=True)

    if df:
        return r2_df
    else:
        r2_mx = np.array(r2_df.pivot(index="Sample", columns="View", values="R2"))
        return r2_mx


def project_data(
    model,
    data,
    view: Union[str, int] = None,
    factors: Union[int, List[int], str, List[str]] = None,
    df: bool = False,
    feature_intersection: bool = False,
):
    """
    Project new data onto the factor space of the model.
    For the projection, a pseudo-inverse of the weights matrix is calculated
    and its product with the provided data matrix is calculated.
    Parameters
    ----------
    data
        Numpy array or Pandas DataFrame with the data matching the number of features
    view : optional
        A view of the model to consider (first view by default)
    factors : optional
        Indices of factors to use for the projection (all factors by default)
    """
    if view is None:
        view = 0
    view = model._check_views([view])[0]
    factor_indices, factors = model._check_factors(factors)

    # Calculate the inverse of W
    winv = np.linalg.pinv(model.get_weights(model=model, views=view, factors=factors))
    winv = np.concatenate(winv)

    # Find feature intersection to match the dimensions
    if feature_intersection:
        if data.shape[1] != model.shape[1] and isinstance(data, pd.DataFrame):
            fs_common = np.intersect1d(data.columns.values, model.features[view])
            data = data.loc[:, fs_common]

            # Get indices of the common features in the original data
            f_sorted = np.argsort(model.features[view])
            fs_common_pos = np.searchsorted(model.features[view][f_sorted], fs_common)
            f_indices = f_sorted[fs_common_pos]

            winv = winv[:, f_indices]
            warnings.warn(
                "Only {} features are matching between two datasets of size {} (original data) and {} (projected data).".format(
                    fs_common.shape[0], model.shape[1], data.shape[1]
                )
            )

    # Predict Z for the provided data
    zpred = np.dot(data, winv.T)

    if df:
        zpred = pd.DataFrame(zpred)
        zpred.columns = factors
        if isinstance(data, pd.DataFrame):
            zpred.index = data.index
    return zpred


def get_views_contributions(model, scaled: bool = True):
    """
    Project new data onto the factor space of the model.
    For the projection, a pseudo-inverse of the weights matrix is calculated
    and its product with the provided data matrix is calculated.
    Parameters
    ----------
    scaled : bool, optional
        Whether to scale contributions scores per sample
        so that they sum up to 1 (True by default)
    Returns
    -------
    pd.DataFrame
        Dataframe with view contribution scores, samples in rows and views in columns
    """
    z = model.get_factors()
    z = np.abs(z) / np.max(np.abs(z), axis=0)

    factors_ordered = model.get_r2().Factor.drop_duplicates().values

    contributions = []
    r2 = model.get_r2()
    for g in model.groups:
        r2_g = r2[r2.apply(lambda x: x.Group == g, axis=1)]
        z_g_indices = model.metadata.group.values == g
        z_g = z[z_g_indices]
        # R2 per factor
        r2_per_factor = r2_g.pivot(index="Factor", columns="View", values="R2").loc[
            factors_ordered, model.view_names
        ]
        # R2 per view
        r2_per_view = np.array(model.model["variance_explained"]["r2_total"][g])
        # Z x R2 per factor
        view_contribution = np.dot(z_g, r2_per_factor) / r2_per_view
        if scaled:
            # Scale contributions to sum to 1
            view_contribution = (
                view_contribution / view_contribution.sum(axis=1)[:, None]
            )
        view_contribution = pd.DataFrame(
            view_contribution,
            index=model.metadata.index[z_g_indices],
            columns=r2_per_factor.columns,
        )
        contributions.append(view_contribution)

    view_contribution = pd.concat(contributions, axis=0)

    return view_contribution


# def get_interpolated_factors(
#     model,
#     groups: Union[str, int, List[str], List[int]] = None,
#     factors: Optional[Union[int, List[int], str, List[str]]] = None,
#     df: bool = False,
#     df_long: bool = False,
#     concatenate_groups: bool = True,
#     scale: bool = False,
#     absolute_values: bool = False,
# ):
#     """
#     Get the matrix with interpolated factors.
#     If df_long is False, a dictionary with keys ("mean", "variance") is returned
#     with NumPy arrays (df=False) or DataFrames (df=True) as values.
#     If df_long is True, a DataFrame with columns ("new_value", "factor", "mean", "variance")
#     is returned.
#     Parameters
#     ----------
#     groups : optional
#         List of groups to consider
#     factors : optional
#         Indices of factors to consider
#     df : optional
#         Boolean value if to return mean and variance matrices as (wide) DataFrames
#         (can be superseded by df_long=True)
#     df_long : optional
#         Boolean value if to return a single long DataFrame
#         (supersedes df=False and concatenate_groups=False)
#     concatenate_groups : optional
#         If concatenate Z matrices (True by default, can be superseded by df_long=True)
#     scale : optional
#         If return values scaled to zero mean and unit variance
#         (per factor when concatenated or per factor and per group otherwise)
#     absolute_values : optional
#         If return absolute values for weights
#     """

#     groups = model._check_groups(groups)
#     factor_indices, factors = model._check_factors(factors)

#     z_interpolated = dict()
#     new_values_names = tuple()
#     if model.covariates_names:
#         new_values_names = tuple(
#             [f"{value}_transformed" for value in model.covariates_names]
#         )
#     else:
#         new_values_names = tuple(
#             [
#                 f"new_value{i}"
#                 for i in range(model.interpolated_factors["new_values"].shape[1])
#             ]
#         )

#     for stat in ["mean", "variance"]:
#         # get factors
#         z = list(
#             np.array(model.interpolated_factors[stat][g])[:, factor_indices]
#             for g in groups
#         )

#         # consider transformations
#         for g in range(len(groups)):
#             if not concatenate_groups:
#                 if scale:
#                     z[g] = (z[g] - z[g].mean(axis=0)) / z[g].std(axis=0)
#                 if absolute_values:
#                     z[g] = np.absolute(z[g])
#             if df or df_long:
#                 z[g] = pd.DataFrame(z[g])
#                 z[g].columns = factors
#                 z[g]["group"] = model.groups[g]

#                 if "new_values" in model.interpolated_factors:
#                     new_values = np.array(model.interpolated_factors["new_values"])
#                 else:
#                     new_values = np.arange(z[g].shape[0]).reshape(-1, 1)

#                 new_values = pd.DataFrame(new_values, columns=new_values_names)

#                 z[g] = pd.concat([z[g], new_values], axis=1)

#                 # If groups are to be concatenated (but not in a long DataFrame),
#                 # index has to be made unique per group
#                 new_samples = [
#                     f"{groups[g]}_{'_'.join(value.astype(str))}"
#                     for _, value in new_values.iterrows()
#                 ]

#                 z[g].index = new_samples

#                 # Create an index for new values
#                 z[g]["new_value"] = np.arange(z[g].shape[0])

#         # concatenate views if requested
#         if concatenate_groups:
#             z = pd.concat(z) if df or df_long else np.concatenate(z)
#             if scale:
#                 z = (z - z.mean(axis=0)) / z.std(axis=0)
#             if absolute_values:
#                 z = np.absolute(z)

#         # melt DataFrames
#         if df_long:
#             if not concatenate_groups:  # supersede
#                 z = pd.concat(z)
#             z = (
#                 z.rename_axis("new_sample", axis=0)
#                 .reset_index()
#                 .melt(
#                     id_vars=["new_sample", *new_values_names, "group"],
#                     var_name="factor",
#                     value_name=stat,
#                 )
#             )

#         z_interpolated[stat] = z

#     if df_long:
#         z_interpolated = (
#             z_interpolated["mean"]
#             .set_index(["new_sample", *new_values_names, "group", "factor"])
#             .merge(
#                 z_interpolated["variance"],
#                 on=("new_sample", *new_values_names, "group", "factor"),
#             )
#         )

#     return z_interpolated


# def get_group_kernel(model):
#     model_groups = False
#     if (
#         model.options
#         and "smooth" in model.options
#         and "model_groups" in model.options["smooth"]
#     ):
#         model_groups = bool(model.options["smooth"]["model_groups"].item().decode())

#     kernels = list()
#     if not model_groups or model.ngroups == 1:
#         Kg = np.ones(shape=(model.n_factors, model.ngroups, model.ngroups))
#         return Kg
#     else:
#         if model.training_stats and "Kg" in model.training_stats:
#             return model.training_stats["Kg"]
#         else:
#             raise ValueError(
#                 "No group kernel was saved. Specify the covariates and train the MEFISTO model with the option 'model_groups' set to True."
#             )
