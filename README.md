![tests](https://github.com/timtreis/T2022B_MSc_thesis/actions/workflows/pythonpackage.yml/badge.svg)
[![codecov](https://codecov.io/gh/timtreis/T2022B_MSc_thesis/branch/main/graph/badge.svg?token=VO0A3UCIH7)](https://codecov.io/gh/timtreis/T2022B_MSc_thesis)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

# MFMF

Presentation -> https://docs.google.com/presentation/d/1N1i9MGLHOXJYNZMwxCL7KUqSHyVzfkig8iVraccoiXM/edit?usp=sharing

## Relevant literature

| Tool        | PDF                                                                  |
| ----------- | -------------------------------------------------------------------- |
| Spatial NMF | https://arxiv.org/pdf/2110.06122.pdf                                 |
| ZINB-WaVE   | https://www.nature.com/articles/s41467-017-02554-5.pdf               |
| GLM-PCA     | https://link.springer.com/content/pdf/10.1186/s13059-019-1861-6.pdf  |
| NMF         | https://www.pnas.org/content/pnas/101/12/4164.full.pdf               |
| sparse PCA  | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2697346/pdf/kxp008.pdf  |
| MOFA        | https://www.embopress.org/doi/full/10.15252/msb.20178124             |
| MEFISTO     | https://www.nature.com/articles/s41592-021-01343-9.pdf               |
| FISHFactor  | https://www.biorxiv.org/content/10.1101/2021.11.04.467354v1.full.pdf |

---

## Priors

MFMF implementes different published priors:

- ### Horseshoe as in [Carvalho et al. (2009)](http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf) -> $(\beta_i | \lambda_i, \tau) \sim N(0, \lambda_i^2 \tau^2)$

  Carvalho et al. define $\lambda_i \sim C^+(0,1)$ as the local shrinkage parameter and $\tau \sim C^+(0,1)$ as the global shrinkage parameter. Since, in the paper, the authors note $\lambda_i^2 \tau^2$ as the variance but Pyro requires the standard deviation, we use $\lambda_i \tau$ as the scale parameter.

- ### Regularised Horseshoe as in [Piironen and Vehtari, 2017](https://arxiv.org/pdf/1707.01694.pdf) -> $(\beta_i | \lambda_i, \tau, c) \sim N(0, \tau^2 \frac{c^2\lambda_i^2}{c^2~+~\tau^2\lambda_i^2})$

  Piironen and Vehtari extend the horsehoe prior with a constant $c^2 \sim 1/Gamma()$ which is used to regularise the prior. When $\tau^2\lambda^2 << c^2$, the prior approaches the standard horseshoe prior. When $\tau^2\lambda^2 >> c^2$, the prior will regularise even the largest coefficients as a Gaussian slab with variance $c^2$. Empirically, we sample $c^2 \sim 1/Gamma(0.5, 0.5)$.

  scale = $\sqrt{\tau^2 \frac{c^2\lambda_i^2}{c^2+\tau^2\lambda_i^2}}\rightarrow\sqrt{\tau^2} \cdot \sqrt{ \frac{c^2\lambda_i^2}{c^2+\tau^2\lambda_i^2}}\underrightarrow{\lambda, \tau, c > 0}$

- ### Spike'n'slab as implemented in [Argelaguet et al. (2018)](https://www.embopress.org/doi/full/10.15252/msb.20178124) -> $N(w^m_{d,k} | 0, 1/\alpha_k^m)~Ber(s^m_{d,k} | \theta_k^m)$

  Argelaguet et al. model the Dirac impulse as $\theta_k^m \sim Beta(a_0^\theta, b_0^\theta)$ and the contribution of the spike as $\alpha_k^m \sim Gamma(a_0^\alpha, b_0^\alpha)$. Empirically, we sample $\theta_k^m \sim Beta(1, 1)$ and $\alpha_k^m \sim Gamma(0.00001, 0.00001)$ to obtain uninformative priors.

---

## Code quicklinks

| Tool       | Repo                                 |
| ---------- | ------------------------------------ |
| FISHFactor | https://github.com/bioFAM/FISHFactor |
| MEFISTO    | https://github.com/bioFAM/MOFA2      |
| mofapy2    | https://github.com/bioFAM/mofapy2    |
| MOFA       | https://github.com/bioFAM/MOFA       |

---

## Q&A

<details>
  <summary>Do I put my "Is function input valid" inside my function or the respective setter?</summary>

> "Note 2: Avoid using properties for computationally expensive operations; the attribute notation makes the caller believe that access is (relatively) cheap." - PEP8

</details>
