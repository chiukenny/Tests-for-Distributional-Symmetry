# Hypothesis Tests for Distributional Symmetry

This repository contains the code used to generate the experiment results in [Non-parametric Hypothesis Tests for Distributional Group Symmetry](https://arxiv.org/abs/2307.15834).

* To reproduce the experiment results, execute the command `julia --threads 1 run_experiments.jl`.
* To generate the plots in the manuscript, execute the command `julia make_plots.jl`.

See the headers of `run_experiments.jl` and `make_plots.jl` for additional execution and configuration options.

---
#### Organization

This GitHub repo is organized as follows:

* `/data/`: raw datasets and data cleaning scripts
* `/experiments/`: self-contained experiment scripts that can be run in parallel
* `/outputs/`: experiment outputs and plots
* `/src/`: implementations of tests, experiment engine, and general functions