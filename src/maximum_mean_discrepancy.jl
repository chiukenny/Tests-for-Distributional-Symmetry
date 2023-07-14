## Implementation of MMD-based tests including
##     1. Monte Carlo MMD
##     2. Nystrom MMD


using Distributions
using Base.Threads
include("util.jl")
include("groups.jl")
include("resampler.jl")


# Shared functions
# ----------------

abstract type AbstractMMD end


# Updates the kernel matrix
function set_kernel_mat(test::AbstractMMD, f_kernel_mat::Function)
    test.f_kernel_mat = f_kernel_mat
end
function set_kernel_mat(test::AbstractMMD, params::Vector{Float64})
    test.k_params[:x] = params
end


# Initializes the test
function initialize(test::AbstractMMD, data_tr::AbstractData)
    if test.presample_actions
        # Pre-sample group actions for reuse across iterations/simulations
        M = test.GS.M
        if typeof(data_tr) == OneSampleData
            n = size(data_tr.x, 2)
            test.g_actions = [Array{Any}(undef,2,M,n)]
            view_actions = view(test.g_actions[1], 1:2, 1:M, 1:n)
            @threads for iter in eachindex(view_actions)
                i = iter[1]
                j = iter[2]
                k = iter[3]
                test.g_actions[1][i,j,k] = test.GS.f_sample()
            end
        else
            n1 = size(data_tr.x1, 2)
            n2 = size(data_tr.x2, 2)
            test.g_actions = [Array{Any}(undef,2,M,n1), Array{Any}(undef,2,M,n2)]
            view_actions1 = view(test.g_actions[1], 1:2, 1:M, 1:n1)
            @threads for iter in eachindex(view_actions1)
                i = iter[1]
                j = iter[2]
                k = iter[3]
                test.g_actions[1][i,j,k] = test.GS.f_sample()
            end
            view_actions2 = view(test.g_actions[2], 1:2, 1:M, 1:n2)
            @threads for iter in eachindex(view_actions2)
                i = iter[1]
                j = iter[2]
                k = iter[3]
                test.g_actions[2][i,j,k] = test.GS.f_sample()
            end
        end
    end
    
    # Use the kernel given with the data if kernel has not been set
    if !haskey(test.k_params, :x)
        σ = typeof(data_tr)==OneSampleData ? data_tr.σx : data_tr.σ
        if length(σ) == 1
            set_kernel_mat(test, [σ])
        else
            set_kernel_mat(test, σ)
        end
    end
end


# Runs the one-sample MMD test
function run_test(test::AbstractMMD, data::OneSampleData, α::Float64)
    if typeof(test)==MMD && !test.invariance
        error("Standard one-sample MMD test not implemented")
    elseif size(data.y,1) > 0
        error("One-sample equivariance test not implemented")
    end
    test_stat = test_statistic(test, data.x)
    
    # Estimate the p-value
    gx = transform_all(test.GS, data.x)
    bvals = resample(test, gx)
    p = resampler_pvalue(test, test_stat, bvals)
    
    return TestSummary(test.name, test_stat, p<=α, pvalue=p)
end


# Runs the two-sample MMD test
function run_test(test::AbstractMMD, data::TwoSampleData, α::Float64)
    if typeof(test) == NMMD
        error("Two-sample Nystrom MMD not implemented")
    end
    test_stat = test_statistic(test, data.x1, data.x2)
    
    # Estimate the p-value
    bvals = resample(test, data.x1, data.x2)
    p = resampler_pvalue(test, test_stat, bvals)
    
    return TestSummary(test.name, test_stat, p<=α, pvalue=p)
end


# Monte Carlo MMD test
# --------------------

mutable struct MMD <: AbstractMMD
    name::String                            # Test name for outputs
    GS::GroupSampler                        # Group being tested
    presample_actions::Bool                 # Pre-sample group actions?
    g_actions::Array{Array{Any,3}}          # Pre-sampled group actions
    RS::Resampler                           # Resampling method
    f_kernel_mat::Function                  # Kernel matrix
    k_params::Dict{Symbol,Vector{Float64}}  # Kernel parameters
    invariance::Bool                        # If false, run standard two-sample test
    
    function MMD(name=""; GS=GroupSampler(), presample_actions=true, g_actions=[Array{Any}(undef,0,0,0)],
                 RS=Resampler(), f_kernel_mat=Gaussian_kernel_mat, k_params=Dict(), invariance=true)
        return new(name, GS, presample_actions, g_actions, RS, f_kernel_mat, k_params, invariance)
    end
end


# Computes the one-sample test statistic
function test_statistic(test::MMD, x::Matrix{Float64})
    f_K = (x1,x2) -> test.f_kernel_mat(x1,x2,test.k_params[:x])
    d, n = size(x)
    
    # Compute the untransformed kernel matrix
    K = f_K(x, x)
    mmd0 = (sum(K)-sum(diag(K))) / (n*(n-1))
    
    # Pre-compute one set of the transformed kernel matrices
    GS = test.GS
    gx = Array{Float64}(undef, GS.M, d, n)
    @threads for m in 1:GS.M
        gx[m,:,:] = @views test.presample_actions ? transform_each(GS,x,test.g_actions[1][1,m,:]) : transform_all(GS,x)
    end
    
    # Compute the test statistic
    mmd = Vector{Float64}(undef, GS.M)
    @threads for m in 1:GS.M
        # Compute the other set of the transformed kernel matrices
        hx = @views test.presample_actions ? transform_each(GS,x,test.g_actions[1][2,m,:]) : transform_all(GS,x)
        mmd[m] = @views - mean(f_K(x,gx[m,:,:])) - mean(f_K(x,hx))
        
        # Compute the cross-term kernel matrix
        for l in 1:GS.M
            K = @views f_K(gx[l,:,:], hx)
            mmd[m] += (sum(K)-sum(diag(K))) / (n*(n-1)*GS.M)
        end
    end
    return mmd0 + mean(mmd)
end


# Computes the two-sample test statistic
function test_statistic(test::MMD, x1::Matrix{Float64}, x2::Matrix{Float64})
    f_K = (x1,x2) -> test.f_kernel_mat(x1,x2,test.k_params[:x])
    d, n1 = size(x1)
    n2 = size(x2, 2)
    
    if !test.invariance
        # Perform the standard two-sample test (Gretton, 2012)
        K1 = f_K(x1, x1)
        K2 = f_K(x2, x2)
        return (sum(K1)-sum(diag(K1)))/(n1*(n1-1)) + (sum(K2)-sum(diag(K2)))/(n2*(n2-1)) - 2*mean(f_K(x1,x2))
    end
    
    # Pre-compute one set of the transformed kernel matrices
    GS = test.GS
    gx1 = Array{Float64}(undef, GS.M, d, n1)
    gx2 = Array{Float64}(undef, GS.M, d, n2)
    @views begin
        @threads for m in 1:GS.M
            gx1[m,:,:] = test.presample_actions ? transform_each(GS,x1,test.g_actions[1][1,m,:]) : transform_all(GS,x1)
            gx2[m,:,:] = test.presample_actions ? transform_each(GS,x2,test.g_actions[2][1,m,:]) : transform_all(GS,x2)
        end
    end
        
    # Compute the test statistic
    mmd = zeros(GS.M)
    @views begin
        @threads for m in 1:GS.M
            # Compute the other set of the transformed kernel matrices
            hx1 = test.presample_actions ? transform_each(GS,x1,test.g_actions[1][2,m,:]) : transform_all(GS,x1)
            hx2 = test.presample_actions ? transform_each(GS,x2,test.g_actions[2][2,m,:]) : transform_all(GS,x2)
            for l in 1:GS.M
                K = f_K(gx1[l,:,:], hx1)
                mmd[m] += (sum(K)-sum(diag(K))) / (n1*(n1-1))
                K = f_K(gx2[l,:,:], hx2)
                mmd[m] += (sum(K)-sum(diag(K)))/(n2*(n2-1)) - mean(f_K(gx1[l,:,:],hx2)) - mean(f_K(gx2[l,:,:],hx1))
            end
        end
    end
    return sum(mmd) / GS.M^2
end


# Nystrom MMD test
# ----------------

mutable struct NMMD <: AbstractMMD
    name::String                            # Test name for outputs
    J::Int64                                # Number of points to sample for Nystrom approximation
    GS::GroupSampler                        # Group being tested
    presample_actions::Bool                 # Pre-sample group actions?
    g_actions::Array{Array{Any,3}}          # Pre-sampled group actions
    RS::Resampler                           # Resampling method
    f_kernel_mat::Function                  # Kernel matrix
    k_params::Dict{Symbol,Vector{Float64}}  # Kernel parameters
    
    function NMMD(name=""; J=1,
                  GS=GroupSampler(), presample_actions=true, g_actions=[Array{Any}(undef,1,1,1)],
                  RS=Resampler(), f_kernel_mat=Gaussian_kernel_mat, k_params=Dict())
        return new(name, J, GS, presample_actions, g_actions, RS, f_kernel_mat, k_params)
    end
end


# Computes the one-sample test statistic
function test_statistic(test::NMMD, x::Matrix{Float64})
    f_K = (x1,x2) -> test.f_kernel_mat(x1,x2,test.k_params[:x])
    d, n = size(x)
    
    # Subsample data and compute the non-transformed term
    t = x[:, sample(1:n,test.J,replace=true)]
    Kt = f_K(t,t)
    ψ = mean(pinv(Kt)*f_K(t,x), dims=2)
    mmd0 = (ψ'*Kt*ψ)[]
    
    # Pre-compute one set of the transformed kernel matrices
    GS = test.GS
    gt = Array{Float64}(undef, GS.M, d, test.J)
    gψ = Matrix{Float64}(undef, GS.M, test.J)
    @views begin
        @threads for m in 1:GS.M
            gx = test.presample_actions ? transform_each(GS,x,test.g_actions[1][1,m,:]) : transform_all(GS,x)
            gt[m,:,:] = gx[:, sample(1:n,test.J,replace=true)]
            gψ[m,:] = mean(pinv(f_K(gt[m,:,:],gt[m,:,:])) * f_K(gt[m,:,:],gx), dims=2)
        end
    end
    
    # Compute the MMD
    mmd = Vector{Float64}(undef, GS.M)
    @views begin
        @threads for m in 1:GS.M
            # Compute the other set of the transformed kernel matrices
            hx = test.presample_actions ? transform_each(GS,x,test.g_actions[1][2,m,:]) : transform_all(GS,x)
            ht = hx[:, sample(1:n,test.J,replace=true)]
            hψ = mean(pinv(f_K(ht,ht)) * f_K(ht,hx), dims=2)
            mmd[m] = -(ψ'*(f_K(t,gt[m,:,:])*gψ[m,:] + f_K(t,ht)*hψ))[] / GS.M

            # Compute the cross-term kernel matrices
            val = zeros(1,test.J)
            for l in 1:GS.M
                val = val + gψ[l,:]'*f_K(gt[l,:,:],ht)
            end
            mmd[m] += dot(val,hψ)[] / GS.M^2
        end
    end
    return mmd0 + sum(mmd)
end