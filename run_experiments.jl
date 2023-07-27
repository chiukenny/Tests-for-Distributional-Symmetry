## USAGE
## -----
##
## From the command line, execute the following command to run all experiments in './experiments/':
##     julia --threads 1 run_experiments.jl
##
## To run only ./experiments/<script>.jl, execute the following command:
##     julia --threads 1 run_experiments.jl <script>.jl
##
## Edit the 'Experiment settings' and 'Paths' variables below as necessary


# Experiment settings
use_raw_data = true  # Start from raw data for SWARM/LHC/TQT? Only need to do so if running for the first time
train_kernel = true  # Train kernel bandwidths? Only need to do so if running for the first time with save_kernel=true
save_kernel = true   # Save trained kernels?

# Paths
dir_src = "./src/"               # Source code directory
dir_dat = "./data/"              # Raw data directory
dir_exp = "./experiments/"       # Experiment directory
dir_out = "./outputs/"           # Output directory
dir_out_dat = "./outputs/data/"  # Cleaned data directory


# -------------------


println("Loading packages and modules")

using Base.Threads
using Statistics
using Distributions
using LinearAlgebra
using Random
using InvertedIndices
using DataFrames
using JLD2
using CSV
using HDF5
import H5Zblosc

include(dir_src * "global_variables.jl")                 # Global variables
include(dir_src * "util.jl")                             # Shared functions
include(dir_src * "groups.jl")                           # Groups and related functions
include(dir_src * "resampler.jl")                        # Resampling functions
include(dir_src * "maximum_mean_discrepancy.jl")         # MMD test
include(dir_src * "cramer_wold.jl")                      # CW test
include(dir_src * "transformation_2sample.jl")           # 2sMMD test
include(dir_src * "kernel_conditional_independence.jl")  # KCI test
include(dir_src * "conditional_permutation.jl")          # CP test
include(dir_src * "experiment_helpers.jl")               # Experiment helper functions


# Clean and save real data
if use_raw_data
    println("Cleaning data")
    include(dir_dat * "SWARM_data.jl")
    include(dir_dat * "LHC_data.jl")
    include(dir_dat * "TQT_data.jl")
end


# Run experiment(s)
if isempty(ARGS)
    println("Running all experiments in folder '$(dir_exp)'\n")
    exps = readdir(dir_exp)
    n_exps = length(exps)
    for i in 1:n_exps
        exp = exps[i]
        println("($(i)/$(n_exps)) Running $(exp)")
        local t = @elapsed include(dir_exp * exp)
        println("($(i)/$(n_exps)) Experiment $(exp) completed in $(ceil(Int,t)) seconds\n")
    end
else
    println("Running $(ARGS[1])")
    t = @elapsed include(dir_exp * ARGS[1])
    println("Experiment $(ARGS[1]) completed in $(ceil(Int,t)) seconds\n")
end
println("Finished running experiments")