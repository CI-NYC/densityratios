# Density Ratios

Python package and R package to estimate density ratios in causal inference using Bregman divergences.

## Python package

For development we use [Pixi](https://pixi.sh/latest/installation/) package manager, which must be installed manually.
You can install the `density_ratio` package in development mode using:

```bash
pixi run pre-commit-install
pixi run postinstall
pixi run test
```

## R package

The R package is a light weight package which provides a `density_ratio_family` function
for density ratio modeling.
To install and view documentation run the following in R

```r
devtools::install('R-package')

library(densityratios)

?density_ratio_family
```

## Related Software

* python KLIEP [github.com/srome/pykliep](https://github.com/srome/pykliep)
* python uLSIF [github.com/hoxo-m/densratio_py](https://github.com/hoxo-m/densratio_py)
* python uLSIF [github.com/JohnYKiyo/density_ratio_estimation](https://github.com/JohnYKiyo/density_ratio_estimation)
* python [github.com/ermongroup/dre-infinity](https://github.com/ermongroup/dre-infinity)
* python [github.com/ermongroup/f-dre](https://github.com/ermongroup/f-dre).
* matlab (https://www.ms.k.u-tokyo.ac.jp/sugi/software.html).
* R (using C++) uLSIF, KLIEP, KMM, LSHH, spectral methods [github.com/thomvolker/densityratio](https://github.com/thomvolker/densityratio) also available on [CRAN]( https://CRAN.R-project.org/package=densityratio).
* R uLSIF and KLIEP [github.com/hoxo-m/densratio](https://github.com/hoxo-m/densratio) also on [CRAN](https://CRAN.R-project.org/package=densratio)
