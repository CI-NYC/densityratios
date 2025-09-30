# Replication code

This directory contains replication code to accompany the paper:
Learning density ratios in causal inference using generalized Riesz losses.

The directory is organized as:

* `models` directory, which contains model and augmentation parameters.
* `simulations.py` script to generate data and run simulations.

Simulations can be run via the command line as

```bash
pixi run paper simulations --model-file binary.yaml
pixi run paper simulations --model-file shift.yaml
pixi run paper simulations --model-file stabilized_weights_mc.yaml
pixi run paper simulations --model-file stabilized_weights_mc_shuffle.yaml
pixi run paper simulations --model-file stabilized_weights_mc_derange.yaml
```

To see additional options run `pixi run paper simulations --help`.
Example plots of predicted vs. fitted values for a single run can be created using the `plot` command, e.g.

```bash
pixi run paper plot --model-file shift.yaml --seed 12345
```
