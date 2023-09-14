<p align="center">
    <img src="https://github.com/bioFAM/cellij/blob/main/docs/_static/logo_black.png" alt="logo" width="500"/>
</p>

# Cellij

[![tests](https://github.com/bioFAM/cellij/actions/workflows/package.yml/badge.svg)](https://github.com/bioFAM/cellij/actions/workflows/package.yml)
[![codecov](https://codecov.io/github/bioFAM/cellij/branch/main/graph/badge.svg?token=IJ4UMMUIW9)](https://codecov.io/github/bioFAM/cellij)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

Cellij (pronounced as "zillīj", derived from [Zellij](https://en.wikipedia.org/wiki/Zellij): a style of mosaic tilework made from individually hand-chiseled tile pieces) is a versatile factor analysis framework for rapidly building and training a wide range of factor analysis models on multi-omics data. Cellij builds upon a Bayesian factor analysis skeleton that is designed to provide a wide-ranging customisability at all levels, ranging from likelihoods and optimisation procedures to sparsity-inducing priors. 

<p align="center">
    <img src="https://github.com/bioFAM/cellij/blob/main/docs/_static/figure1_black.png" alt="schematic" width="750"/>
</p>

Cellij is designed for rapid prototyping of custom factor analysis models, allowing users to efficiently define new models in an iterative fashion. The following code snippet shows an example how to setup and train a model with a predefined sparsity prior.
```python
mdata = cellij.Importer().load_CLL()

# 1. We create a new Factor Analysis model
model = cellij.FactorModel(n_factors=10)

# 2. We add an MuData object to the model
model.add_data(mdata)

# 3. We can add some options if we wish
model.set_model_options(
    weight_priors={
        "drugs": "Horseshoe",
        "methylation": "Horseshoe",
        "mrna": "Horseshoe",
    },
)

# 4. We train the model
model.fit(epochs=10000)
```
For basic tutorials on real-world data, please have a look at [our notebook repository]([https://github.com/bioFAM/cellij/blob/main/notebooks/basic_example_mofa.ipynb](https://github.com/bioFAM/cellij-notebooks/tree/main)).


## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install cellij:

<!--
1) Install the latest release of `cellij` from `PyPI <https://pypi.org/project/cellij/>`_:

```sh
pip install cellij
```
-->

1. Install the latest development version:

```sh
pip install git+https://github.com/bioFAM/cellij.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation
If you use Cellij, please consider citing:
```
@proceedings{rohbeckcellij,
  author    = {Rohbeck, Martin and Qoku, Arber and Treis, Tim and Theis, Fabian J and Velten, Britta and Buettner, Florian and Stegle, Oliver},
  title     = {Cellij: A Modular Factor Model Framework for Interpretable and Accelerated Multi-Omics Data Integration},
  series    = {ICML Workshop on Computational Biology},
  year      = {2023},
  url       = {https://icml-compbio.github.io/2023/papers/WCBICML2023_paper124.pdf}
}
```

## Docs and Changelog

[changelog]: https://cellij.readthedocs.io/latest/changelog.html
[link-docs]: https://cellij.readthedocs.io
[link-api]: https://cellij.readthedocs.io/latest/api.html

