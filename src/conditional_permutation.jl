## Implementation of the conditional permutation test (Berrett et al., 2019)
## with kernel conditional density estimation


using Base.Threads
using Distributions
using LinearAlgebra
include("util.jl")
include("groups.jl")


mutable struct CP
    name::String                            # Test name for outputs
    S::Int64                                # Number of steps in CP
    B::Int64                                # Number of resampling iterations
    f_T::Function                           # Statistic for CP
    f_kernel_mat_y::Function                # Kernel matrix for Y
    f_kernel_mat_z::Function                # Kernel matrix for Z (maximal invariant)
    k_params::Dict{Symbol,Vector{Float64}}  # Kernel parameters
    
    function CP(name=""; S=50, B=200, f_T=f_nothing,
                f_kernel_mat_y=Gaussian_kernel_mat, f_kernel_mat_z=Gaussian_kernel_mat, k_params=Dict())
        return new(name, S, B, f_T, f_kernel_mat_y, f_kernel_mat_z, k_params)
    end
end


# Updates the kernel matrix
function set_kernel_mat(test::CP, f_kernel_mat::Function, data::Symbol)
    if data == :y
        test.f_kernel_mat_y = f_kernel_mat
    elseif data == :z
        test.f_kernel_mat_z = f_kernel_mat
    else
        error("Argument data=$(data) not supported")
    end
end
function set_kernel_mat(test::CP, params::Vector{Float64}, data::Symbol)
    if data in (:y,:z)
        test.k_params[data] = params
    else
        error("Argument data=$(data) not supported")
    end
end


# Optimizes the kernel bandwidths via grid search
function optimize_kernel(test::CP, f_sample_H0_data::Function; f_sample_H1_data::Function=f_nothing,
                         α::Float64=0.05, N::Int64=100, n::Int64=50,
                         σys::Vector{Float64}=Float64[], σzs::Vector{Float64}=Float64[])
    σs = zeros(2)
    KS = KernelSummary(σs)
    
    # Initialize grid search
    n_y = length(σys)
    opt_y = n_y > 0
    n_y = max(n_y, 1)
    
    n_z = length(σzs)
    opt_z = n_z > 0
    n_z = max(n_z, 1)
    
    # Give each thread its own copy of the test
    thread_tests = []
    for i in 1:nthreads()
        push!(thread_tests, deepcopy(test))
    end
    
    # Estimate levels
    sig = zeros(n_y, n_z)
    view_sig = view(sig, 1:n_y, 1:n_z)
    for _ in 1:N
        data0 = f_sample_H0_data(n)
        @threads for iter in eachindex(view_sig)
            i = iter[1]
            j = iter[2]
            gsTest = thread_tests[threadid()]
            if opt_y
                set_kernel_mat(gsTest, [σys[i]], :y)
            end
            if opt_z
                set_kernel_mat(gsTest, [σzs[j]], :z)
            end
            _, p = test_statistic(gsTest, data0.x, data0.y, data0.z)
            sig[i,j] += p <= α
        end
    end
    sig = sig ./ N
    KS.info[:sig] = sig
    
    # Estimate powers
    if f_sample_H1_data != f_nothing
        pow = zeros(n_y, n_z)
        for _ in 1:N
            data1 = f_sample_H1_data(n)
            @threads for iter in eachindex(view_sig)
                i = iter[1]
                j = iter[2]
                if sig[i,j] > α+0.05  # Allow some wiggle room
                    continue
                end
                gsTest = thread_tests[threadid()]
                if opt_y
                    set_kernel_mat(gsTest, [σys[i]], :y)
                end
                if opt_z
                    set_kernel_mat(gsTest, [σzs[j]], :z)
                end
                _, p = test_statistic(gsTest, data1.x, data1.y, data1.z)
                pow[i,j] += p <= α
            end
        end
        if sum(pow) > 0
            inds = findmax(pow)[2]
        end
        KS.info[:pow] = pow ./ N
    end
    if f_sample_H1_data==f_nothing || sum(pow)==0
        inds = findmin( α_loss.(sig,α) )[2]
    end
    
    # Save the optimal bandwidths
    if opt_y
        σs[1] = σys[inds[1]]
        set_kernel_mat(test, [σs[1]], :y)
        KS.grids[:y] = σys
    end
    if opt_z
        σs[2] = σzs[inds[2]]
        set_kernel_mat(test, [σs[2]], :z)
        KS.grids[:z] = σzs
    end
    return KS
end


# Initializes the test
function initialize(test::CP, data_tr::AbstractData)
    # Kernels should be set for KCDE but use the kernel given with the data as a last resort
    if !haskey(test.k_params, :y)
        σy = data_tr.σy
        set_kernel_mat(test, [σy], :y)
    end
    if !haskey(test.k_params, :z)
        σz = data_tr.σz
        set_kernel_mat(test, [σz], :z)
    end
end


# Samples a swap based on kernel conditional densities
function KCDE_sampler(i::Int64, j::Int64, K_y::Matrix{Float64}, K_z::Matrix{Float64})
    f_ij = @views K_y[[i,j],:] * K_z[:,[i,j]]
    odds = exp(log(f_ij[1,2]) + log(f_ij[2,1]) - log(f_ij[1,1]) - log(f_ij[2,2]))
    return rand(Bernoulli(odds/(1+odds)))
end


# Computes the CP test statistic (conditional independence of X and Y given Z)
function test_statistic(test::CP, x::AbstractMatrix{Float64}, y::AbstractMatrix{Float64}, z::AbstractMatrix{Float64})
    d, n = size(x)
    n2 = floor(Int64, n/2)
    
    # Compute unpermuted kernel matrices
    K_y = test.f_kernel_mat_y(y, y, test.k_params[:y])
    K_z = test.f_kernel_mat_z(z, z, test.k_params[:z])
    
    # Keep track of overall permutation
    inds = collect(1:n)
    
    # Run initial sequence
    for _ in 1:test.S
        pK_y = K_y[inds, inds]
        pairs = randperm(n)
        @threads for m in 1:n2
            i = pairs[2*(m-1) + 1]
            j = pairs[2*m]
            swap = KCDE_sampler(inds[i], inds[j], pK_y, K_z)
            if swap
                t = inds[i]
                inds[i] = inds[j]
                inds[j] = t
            end
        end
    end
    
    # Run parallel sequences
    Ts = Vector{Float64}(undef, test.B)
    @threads for b in 1:test.B
        bInds = copy(inds)
        for _ in 1:test.S
            bpK_y = K_y[bInds, bInds]
            pairs = randperm(n)
            for m in 1:n2
                i = pairs[2*(m-1) + 1]
                j = pairs[2*m]
                swap = KCDE_sampler(bInds[i], bInds[j], bpK_y, K_z)
                if swap
                    t = bInds[i]
                    bInds[i] = bInds[j]
                    bInds[j] = t
                end
            end
        end
        Ts[b] = test.f_T(x, y[:,bInds], z)
    end
    
    # Compute p-value
    test_stat = sum(Ts .>= test.f_T(x,y,z))
    return test_stat, (1+test_stat) / (1+test.B)
end


# Runs the test
function run_test(test::CP, data::OneSampleData, α::Float64)
    test_stat, p = test_statistic(test, data.x, data.y, data.z)
    return TestSummary(test.name, test_stat, p<=α, pvalue=p)
end