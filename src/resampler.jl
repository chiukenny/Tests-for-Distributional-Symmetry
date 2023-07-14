## Resampling methods for tests that require resampling/bootstrapping


using Random
include("util.jl")
include("groups.jl")


mutable struct Resampler
    B::Int64             # Number of samples to take
    f_sampler::Function  # Sampling method
    n_prop::Float64      # Size of sample relative to original sample (1 being same size)
    function Resampler(;B=200, f_sampler=subsampler, n_prop=1)
        return new(B, f_sampler, n_prop)
    end
end


# Computes set of resampled test statistics for one/two-sample tests
function resample(test::Any, x::AbstractMatrix{Float64})
    n = size(x, 2)
    RS = test.RS
    bvals = zeros(RS.B)
    Threads.@threads for b in 1:RS.B
        bx = RS.f_sampler(test, x)
        bvals[b] = test_statistic(test, bx)
    end
    return bvals
end
function resample(test::Any, x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64})
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    x = hcat(x1, x2)
    n = n1 + n2
    RS = test.RS
    bvals = zeros(RS.B)
    Threads.@threads for b in 1:RS.B
        bx1, bx2 = RS.f_sampler(test, x, n1, n2)
        bvals[b] = test_statistic(test, bx1, bx2)
    end
    return bvals
end


# Computes the p-value given the observed and resampled test statistics
function resampler_pvalue(test::Any, test_stat::Float64, bvals::Vector{Float64})
    return (1 + sum(bvals.>test_stat)) / (1 + test.RS.B)
end


# Common sampling methods
# -----------------------

# Subsamples from data
function subsampler(test::Any, x::AbstractMatrix{Float64})
    n = size(x, 2)
    # Don't use views when subsampling randomly
    return x[:, sample(1:n,ceil(Int64,n*test.RS.n_prop),replace=true)]
end
function subsampler(test::Any, x::AbstractMatrix{Float64}, n1::Int64, n2::Int64)
    n = n1 + n2
    # Don't use views when subsampling randomly
    return x[:,sample(1:n,ceil(Int64,n1*test.RS.n_prop),replace=true)], x[:,sample(1:n,ceil(Int64,n2*test.RS.n_prop),replace=true)]
end


# Transforms the data
function transform_sampler(test::Any, x::AbstractMatrix{Float64})
    return transform_all(test.GS, x)
end


# Transforms the data before subsampling
function transform_subsampler(test::Any, x::AbstractMatrix{Float64})
    return subsampler(test, transform_sampler(test,x))
end


# Subsamples the data before transforming
function subsample_transformer(test::Any, x::AbstractMatrix{Float64})
    return transform_sampler(test, subsampler(test,x))
end