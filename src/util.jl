## Collection of miscellaneous objects and functions shared across tests and experiments


using Random
using Distributions
using Distances
using InvertedIndices
using Base.Threads


# Data
# ----

# Objects for conveniently passing data
abstract type AbstractData end

struct OneSampleData <: AbstractData
    x::Matrix{Float64}                  # X data
    y::Matrix{Float64}                  # Y data
    z::Matrix{Float64}                  # Z (maximal invariant) data
    σx::Union{Float64,Vector{Float64}}  # X median distance
    σy::Float64                         # Y median distance
    σz::Float64                         # Z median distance
    function OneSampleData(;x=Matrix{Float64}(undef,0,0), y=Matrix{Float64}(undef,0,0), z=Matrix{Float64}(undef,0,0),
                           σx=0., σy=0., σz=0.)
        return new(x, y, z, σx, σy, σz)
    end
end

struct TwoSampleData <: AbstractData
    x1::Matrix{Float64}                # Dataset 1
    x2::Matrix{Float64}                # Dataset 2
    σ::Union{Float64,Vector{Float64}}  # Median distance(s)
    function TwoSampleData(;x1=Matrix{Float64}(undef,0,0), x2=Matrix{Float64}(undef,0,0), σ=0.)
        return new(x1, x2, σ)
    end
end

# Samples data from a given distribution and wraps it in an object
function sample_data(n::Int64, P::Distribution; med_heur::Bool=true)
    x = rand(P, n)
    σ = med_heur ? med_dist(x) : 0
    return OneSampleData(x=x, σx=σ)
end
function sample_data(n1::Int64, n2::Int64, P1::Distribution, P2::Distribution; med_heur::Bool=true)
    x1 = rand(P1, n1)
    x2 = rand(P2, n2)
    σ = 0.
    if med_heur
        x = hcat(x1, x2)
        σ = med_dist(x)
    end
    return TwoSampleData(x1=x1, x2=x2, σ=σ)
end


# Test summary
# ------------

# Object for standardizing outputs of tests
mutable struct TestSummary
    name::String             # Test name
    test_statistic::Float64  # Test statistic value
    reject::Bool             # Result of test (1 reject, 0 not reject)
    pvalue::Float64          # p-value of test
    function TestSummary(name, test_statistic, reject; pvalue=NaN)
        return new(name, test_statistic, reject, pvalue)
    end
end

# Object for standardizing outputs of kernel optimization procedures
mutable struct KernelSummary
    σs::Vector{Float64}                  # Optimal bandwidths
    grids::Dict{Symbol,Vector{Float64}}  # Grids to optimize kernels over
    info::Dict{Symbol,Array{Float64}}    # Saved optimization artifacts (rejection rates, etc.) for debugging
    function KernelSummary(σs; grids=Dict(), info=Dict())
        return new(σs, grids, info)
    end
end

# Save and load tests to avoid re-training on subsequent experiment runs
function save_test(file::String, test::Any, summary::KernelSummary)
    obj = Dict(:test=>test, :summary=>summary)
    save_object(file, obj)
end
function load_test(file::String)
    return load_object(file)
end


# General functions
# -----------------


# Generates a random integer
function randInt()
    return rand(1:9999999)
end


# Standardizes a dataset
function standardize(x::AbstractMatrix{Float64})
    return (x .- mean(x,dims=2)) ./ std(x,dims=2)
end


# Applies a small perturbation to a point for procedures that require unique observations
function jitter(x::AbstractVector{Float64})
    d = max(maximum(x)-minimum(x), 1)
    upp = d / (100*(1+d)^2)
    return x .+ rand(length(x)) .* upp
end


# Computes the median distance between points in a sample
function med_dist(x::AbstractMatrix{Float64}; max_n::Int64=100)
    n = size(x, 2)
    m = min(n, max_n)
    s = sample(1:n, m, replace=false)
    len = Int64( m*(m-1)/2 )
    dists = Vector{Float64}(undef, len)
    @threads for i in 1:m
        ind = Int64( (i-1)*(i-2)/2 )
        for j in 1:(i-1)
            dists[ind+j] = @views sqeuclidean(x[:,s[i]], x[:,s[j]])
        end
    end
    return sqrt(median(dists))
end


# Computes a custom loss for determining how close an estimated rejection rate is to a desired rate
function α_loss(rate::Float64, α::Float64)
    if α >= rate
        return α - rate
    end
    return 2 * (rate-α)
end


# Computes the Gaussian kernel matrix for two samples
function Gaussian_kernel_mat(x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64}, params::Vector{Float64})
    σ = params[1]
    s2 = 2 * σ^2
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    K = Matrix{Float64}(undef, n1, n2)
    
    if n1 == n2 && all(x1 .== x2)
        # Compute the kernel matrix for a single sample
        K[1,1] = @views exp(-sqeuclidean(x1[:,1],x2[:,1]) / s2)
        @threads for i in 2:n1
            K[i,i] = K[1,1]
        end
        @threads for i in 1:n1
            for j in 1:(i-1)
                K[i,j] = @views exp(-sqeuclidean(x1[:,i],x2[:,j]) / s2)
                K[j,i] = K[i,j]
            end
        end
        return K
    end
    
    # Compute the kernel matrix between two samples
    view_K = view(K, 1:n1, 1:n2)
    @threads for iter in eachindex(view_K)
        i = iter[1]
        j = iter[2]
        # Note: KDE for CP needs the scaling factor 1/(σ*sqrt(2*π)),
        #       but we only use it for density ratios so the scaling cancels out
        K[i,j] = @views exp(-sqeuclidean(x1[:,i],x2[:,j]) / s2)
    end
    return K
end


# Computes the product Gaussian kernel matrix for the SWARM experiment
function SWARM_Gaussian_kernel_mat(xy1::AbstractMatrix{Float64}, xy2::AbstractMatrix{Float64}, params::Vector{Float64})
    n1 = size(xy1, 2)
    n2 = size(xy2, 2)
    x1 = xy1[1:3, :]
    y1 = reshape(xy1[4, :], 1, n1)
    x2 = xy2[1:3, :]
    y2 = reshape(xy2[4, :], 1, n2)
    return Gaussian_kernel_mat(x1,x2,[params[1]]) .* Gaussian_kernel_mat(y1,y2,[params[2]])
end


# Converts a matrix to a quaternion
function mat_to_quat(X::AbstractMatrix{Float64})
    if typeof(X) != QuatRotation
        X = QuatRotation(X)
    end
    return [X.q.s, X.q.v1, X.q.v2, X.q.v3]
end


# Converts a quaternion to a matrix
function quat_to_mat(q::AbstractVector{Float64})
    return QuatRotation(q)
end


# Computes a matrix product directly from quaternion representations
function quat_prod(q1::AbstractVector{Float64}, q2::AbstractVector{Float64})
    # https://www.songho.ca/opengl/gl_quaternion.html
    return [q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3] - q1[4]*q2[4],
            q1[1]*q2[2] + q1[2]*q2[1] + q1[3]*q2[4] - q1[4]*q2[3],
            q1[1]*q2[3] - q1[2]*q2[4] + q1[3]*q2[1] + q1[4]*q2[2],
            q1[1]*q2[4] + q1[2]*q2[3] - q1[3]*q2[2] + q1[4]*q2[1]]
end


# Computes a characteristic kernel on SO(3) (Fukumizu, 2008)
function SO3_kernel(qX::AbstractVector{Float64}, qY::AbstractVector{Float64})
    X = quat_to_mat(qX)
    Y = quat_to_mat(qY)
    θ = rotation_angle(Y \ X)
    return θ==0 || θ==π ? 0 : π*θ*(π-θ)/(8*sin(θ))
end


# Computes the kernel matrix for SO(3) data
function SO3_kernel_mat(x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64}, params::Vector{Float64})
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    K = Matrix{Float64}(undef, n1, n2)
    
    if n1 == n2 && all(x1 .== x2)
        # Compute the kernel matrix for a single sample
        K[1,1] = @views SO3_kernel(x1[:,1], x1[:,1])
        @threads for i in 2:n1
            K[i,i] = K[1,1]
        end
        @threads for i in 1:n1
            for j in 1:(i-1)
                K[i,j] = @views SO3_kernel(x1[:,i], x2[:,j])
                K[j,i] = K[i,j]
            end
        end
        return K
    end
    
    # Compute the kernel matrix between two samples
    view_K = view(K, 1:n1, 1:n2)
    @threads for iter in eachindex(view_K)
        i = iter[1]
        j = iter[2]
        K[i,j] = @views SO3_kernel(x1[:,i], x2[:,j])
    end
    return K
end


# Computes the indicator kernel matrix for binary data
# https://upcommons.upc.edu/bitstream/handle/2099.1/17172/MarcoVillegas.pdf
function binary_kernel_mat(x1::AbstractMatrix{Float64}, x2::AbstractMatrix{Float64}, params::Vector{Float64})
    n1 = size(x1, 2)
    n2 = size(x2, 2)
    return float(reshape(repeat(x1',n2),n1,n2) .== repeat(x2,n1))
end


# Computes the multiple correlation coefficient between variables x and y
function multiple_correlation(x::AbstractMatrix{Float64}, y::AbstractMatrix{Float64})
    Rxx = cor(x, dims=2)
    Rxy = cor(x, y, dims=2)
    return dot(Rxy, Rxx\Rxy)
end
# Helper function because anonymous functions don't save properly
function multiple_correlation_xyz(x::AbstractMatrix{Float64}, y::AbstractMatrix{Float64}, z::AbstractMatrix{Float64})
    return multiple_correlation(x, y)
end


# Placeholder function for function-typed variables
function f_nothing(args...)
    error("f_nothing($(length(args)) args) called: function has not set")
end