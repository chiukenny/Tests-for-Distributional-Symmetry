## Implementation of the transformation two-sample (baseline) test for invariance


include("util.jl")
include("groups.jl")


mutable struct Transform2S
    name::String  # Test name for outputs
    test          # Core test (currently only compatible with MMD)
    function Transform2S(name, test)
        test = deepcopy(test)
        if typeof(test) != MMD
            error("Transform2S is currently only compatible with MMD")
        end
        # Make sure test mode is set to standard (not invariance)
        if hasfield(typeof(test), :invariance)
            test.invariance = false
        end
        # Make sure subsampling resampling method is used
        if hasfield(typeof(test), :RS)
            test.RS.f_sampler = subsampler
        end
        return new(name, test)
    end
end


# Initializes the test
function initialize(test::Transform2S, data_tr::AbstractData)
    initialize(test.test, data_tr)
end


# Runs the test
function run_test(test::Transform2S, data::OneSampleData, α::Float64)
    GS = test.test.GS
    gx = transform_all(GS, data.x)
    # Compute the median distance if not already provided with the data
    if length(data.σx) > 1 || data.σx > 0
        σ = data.σx
    else
        x = hcat(data.x, gx)
        σ = med_dist(x)
    end
    TS = run_test(test.test, TwoSampleData(x1=data.x,x2=gx,σ=σ), α)
    TS.name = test.name
    return TS
end