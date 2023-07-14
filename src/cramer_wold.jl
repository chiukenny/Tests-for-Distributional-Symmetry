## Implementation of the Cramer-Wold test for invariance (Fraiman, 2022)


using Distributions
using InvertedIndices
using HypothesisTests
using SliceMap
include("util.jl")
include("groups.jl")
include("resampler.jl")


mutable struct CW
    name::String            # Test name for outputs
    GS::GroupSampler        # Group being tested
    RS::Resampler           # Resampling method for non-generator-based test
    multi_projection::Bool  # Use non-generator-based test?
    J::Int64                # Number of random projections for non-generator-based test
    params                  # Cramer-Wold parameters
    
    function CW(name=""; GS=GroupSampler(), RS=Resampler(), multi_projection=true, J=1, params=nothing)
        return new(name, GS, RS, multi_projection, J, params)
    end
end


# Initializes the test
function initialize(test::CW, data_tr::AbstractData)
    test.params = [test.GS.f_sample() for i in 1:test.GS.M]
end


# Computes the CW test statistic
function test_statistic(test::CW, x::Matrix{Float64})
    d, n = size(x)
    n1 = ceil(Int64, n/2)
    n2 = n - n1
    
    GS = test.GS
    if test.multi_projection || GS.f_get_generators==f_nothing
        # Perform the non-generator-based test
        # Sample random projections
        t = rand(MvNormal(zeros(d),1), test.J)
        t = t ./ tmapcols(norm, t)
        
        # Compute the worst-case Kolmogorov-Smirnov statistic
        p1 = tmapcols(jitter, x'*t)
        gs = test.params
        max_ks = 0
        for m in 1:GS.M
            p2 = tmapcols(jitter, transform_all(GS,x,gs[m])'*t)
            for i in 1:test.J
                ks = @views ApproximateTwoSampleKSTest(p1[:,i], p2[:,i])
                max_ks = max(max_ks, sqrt(ks.n_x*ks.n_y/(ks.n_x+ks.n_y))*ks.δ)
            end
        end
        return max_ks
    end
    
    # Perform the generator-based (distribution-free) test
    # Randomly split the data into two sets
    i1 = sample(1:n, n1, replace=false)
    x1 = @views x[:, i1]
    x2 = @views x[:, Not(i1)]
    
    # Sample random projections
    gs = GS.f_get_generators()
    J = length(gs)
    t = rand(MvNormal(zeros(d),1), J)
    t = t ./ tmapcols(norm, t)
    
    # Compute p-values and return the minimum
    pvals = Vector{Float64}(undef, J)
    @views begin
        Threads.@threads for i in 1:J
            g = gs[i]
            p1 = vec(x1' * t[:,i])
            p2 = vec(transform_all(GS,x2,g)' * t[:,i])
            pvals[i] = pvalue(ApproximateTwoSampleKSTest(p1,p2), tail=:right)
        end
    end
    return minimum(pvals)
end


# Runs the test
function run_test(test::CW, data::OneSampleData, α::Float64)
    if size(data.y,1) > 0
        error("One-sample equivariance test not implemented")
    end
    test_stat = test_statistic(test, data.x)
    
    if test.multi_projection || test.GS.f_get_generators==f_nothing
        # If running non-generator-based test, estimate the p-value
        gx = transform_all(test.GS, data.x)
        bvals = resample(test, gx)
        p = resampler_pvalue(test, test_stat, bvals)
        return TestSummary(test.name, test_stat, p<=α, pvalue=p)
    end
    
    return TestSummary(test.name, test_stat, test_stat<=α/test.J, pvalue=test_stat)  # Bonferroni-adjusted p-value
end