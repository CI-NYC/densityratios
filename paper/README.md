# Replication code

This directory contains replication code to accompany the paper:
Learning density ratios in causal inference using generalized Riesz losses.

The directory is organized as:

* `models` directory, which contains model and augmentation parameters.
* `simulations.py` script to generate data and run simulations.

Simulations can be run via the command line as

```bash
pixi run paper run --model-file shift.yaml
pixi run paper run --model-file stabilized_weights_empirical.yaml
pixi run paper run --model-file stabilized_weights_mc.yaml
pixi run paper run --model-file stabilized_weights_quantile.yaml
```

Additionally, example plots for a single run can be created using the command

```bash
pixi run paper plot --model-file shift.yaml --seed 12345
```

To see more options for these commands run e.g. `pixi run paper run --help`.
