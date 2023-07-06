[![Cellij Logo](https://github.com/bioFAM/cellij/blob/main/docs/_static/logo2.png)](https://github.com/bioFAM/cellij)

# cellij

[![tests](https://github.com/bioFAM/cellij/actions/workflows/package.yml/badge.svg)](https://github.com/bioFAM/cellij/actions/workflows/package.yml)
[![codecov](https://codecov.io/github/bioFAM/cellij/branch/main/graph/badge.svg?token=IJ4UMMUIW9)](https://codecov.io/github/bioFAM/cellij)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

Cellij is a versatile factor analysis framework for rapidly building and training a wide range of factor analysis models for multi-omics data. Cellij builds upon a Bayesian FA skeleton that is designed to provide wide-ranging customisability at all levels, ranging from likelihoods and optimisation procedures to sparsity-inducing priors. 

[![Cellij Logo](https://github.com/bioFAM/cellij/blob/main/docs/_static/schematic_view.png)](https://github.com/bioFAM/cellij)

Cellij is designed for rapid prototyping of custom FA models, allowing users to efficiently define new models in an iterative fashion.
```
TODO
```


## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install cellij:

<!--
1) Install the latest release of `cellij` from `PyPI <https://pypi.org/project/cellij/>`_:

```bash
pip install cellij
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/bioFAM/cellij.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

If you use Cellij, please consider citing:
```
@article{bingham2019pyro,
  author    = {Rohbeck, Martin and Qoku, Arber and Treis, Tim and Theis, Fabian J and Velten, Britta and Buettner, Florian and Stegle, Oliver},
  title     = {Cellij: A Modular Factor Model Framework for Interpretable and Accelerated Multi-Omics Data Integration},
  journal   = {XXX},
  year      = {2023},
  url       = {YYY}
}
```

## Docs and Changelog

[changelog]: https://cellij.readthedocs.io/latest/changelog.html
[link-docs]: https://cellij.readthedocs.io
[link-api]: https://cellij.readthedocs.io/latest/api.html

