## Implementation of the kernel conditional independence test (Zhang et al., 2012)


using Distributions
using LinearAlgebra
using Base.Threads
include("util.jl")


mutable struct KCI
    name::String                            # Test name for outputs
    ϵ::Float64                              # Ridge value used in KCI algorithm
    min_λ::Float64                          # Tolerance value used in KCI algorithm
    B::Int64                                # Number of resampling iterations
    f_kernel_mat_x::Function                # Kernel matrix for X
    f_kernel_mat_y::Function                # Kernel matrix for Y
    f_kernel_mat_z::Function                # Kernel matrix for Z (maximal invariant)
    k_params::Dict{Symbol,Vector{Float64}}  # Kernel parameters
    
    function KCI(name=""; ϵ=1e-3, min_λ=1e-5, B=100, k_params=Dict(),
                 f_kernel_mat_x=Gaussian_kernel_mat, f_kernel_mat_y=Gaussian_kernel_mat, f_kernel_mat_z=Gaussian_kernel_mat)
        return new(name, ϵ, min_λ, B, f_kernel_mat_x, f_kernel_mat_y, f_kernel_mat_z, k_params)
    end
end


# Updates the kernel matrix
function set_kernel_mat(test::KCI, f_kernel_mat::Function, data::Symbol)
    if data == :x
        test.f_kernel_mat_x = f_kernel_mat
    elseif data == :y
        test.f_kernel_mat_y = f_kernel_mat
    elseif data == :z
        test.f_kernel_mat_z = f_kernel_mat
    else
        error("Argument data=$(data) not supported")
    end
end
function set_kernel_mat(test::KCI, params::Vector{Float64}, data::Symbol)
    if data in (:x,:y,:z)
        test.k_params[data] = params
    else
        error("Argument data=$(data) not supported")
    end
end


# Optimizes the kernel bandwidths via grid search
function optimize_kernel(test::KCI, f_sample_H0_data::Function; f_sample_H1_data::Function=f_nothing, seed::Int64=randInt(),
                         α::Float64=0.05, N::Int64=100, n::Int64=50,
                         σxs::Vector{Float64}=Float64[], σys::Vector{Float64}=Float64[], σzs::Vector{Float64}=Float64[])
    σs = zeros(3)
    KS = KernelSummary(σs)
    
    # Initialize grid search
    n_x = length(σxs)
    opt_x = n_x > 0
    n_x = max(n_x, 1)
    
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
    Random.seed!(seed + 1000000)
    base_seed = randInt()
    
    # Estimate levels
    sig = zeros(n_x, n_y, n_z)
    view_sig = view(sig, 1:n_x, 1:n_y, 1:n_z)
    for s in 1:N
        # Ensure the simulated data are consistent irrespective of threads at the minimum
        Random.seed!(base_seed + s)
        
        data0 = f_sample_H0_data(n)
        @threads for iter in eachindex(view_sig)
            i = iter[1]
            j = iter[2]
            k = iter[3]
            gsTest = thread_tests[threadid()]
            if opt_x
                set_kernel_mat(gsTest, [σxs[i]], :x)
            end
            if opt_y
                set_kernel_mat(gsTest, [σys[j]], :y)
            end
            if opt_z
                set_kernel_mat(gsTest, [σzs[k]], :z)
            end
            _, p = test_statistic(gsTest, data0.x, data0.y, data0.z)
            sig[i,j,k] += p <= α
        end
    end
    sig = sig ./ N
    KS.info[:sig] = sig
    
    # Estimate powers
    if f_sample_H1_data != f_nothing
        pow = zeros(n_x, n_y, n_z)
        for s in 1:N
            # Ensure the simulated data are consistent irrespective of threads at the minimum
            Random.seed!(base_seed + s)
            
            data1 = f_sample_H1_data(n)
            @threads for iter in eachindex(view_sig)
                i = iter[1]
                j = iter[2]
                k = iter[3]
                if sig[i,j,k] > α+0.05  # Allow some wiggle room
                    continue
                end
                gsTest = thread_tests[threadid()]
                if opt_x
                    set_kernel_mat(gsTest, [σxs[i]], :x)
                end
                if opt_y
                    set_kernel_mat(gsTest, [σys[j]], :y)
                end
                if opt_z
                    set_kernel_mat(gsTest, [σzs[k]], :z)
                end
                _, p = test_statistic(gsTest, data1.x, data1.y, data1.z)
                pow[i,j,k] += p <= α
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
    if opt_x
        σs[1] = σxs[inds[1]]
        set_kernel_mat(test, [σs[1]], :x)
        KS.grids[:x] = σxs
    end
    if opt_y
        σs[2] = σys[inds[2]]
        set_kernel_mat(test, [σs[2]], :y)
        KS.grids[:y] = σys
    end
    if opt_z
        σs[3] = σzs[inds[3]]
        set_kernel_mat(test, [σs[3]], :z)
        KS.grids[:z] = σzs
    end
    return KS
end


# Initializes the test
function initialize(test::KCI, data_tr::AbstractData)
    # Use the kernel given with the data if the kernel has not been set
    if !haskey(test.k_params, :x)
        σx = data_tr.σx
        set_kernel_mat(test, [σx], :x)
    end
    if !haskey(test.k_params, :y)
        σy = data_tr.σy
        set_kernel_mat(test, [σy], :y)
    end
    if !haskey(test.k_params, :z)
        σz = data_tr.σz
        set_kernel_mat(test, [σz], :z)
    end
end


# Computes the KCI test statistic (conditional independence of X and Y given Z)
function test_statistic(test::KCI, x::Matrix{Float64}, y::Matrix{Float64}, z::Matrix{Float64})
    n = size(x, 2)
    
    # Compute the kernel matrices
    Kx = test.f_kernel_mat_x(x, x, test.k_params[:x])
    Ky = test.f_kernel_mat_y(y, y, test.k_params[:y])
    Kz = test.f_kernel_mat_z(z, z, test.k_params[:z])
    Kxz = Kx .* Kz
    
    # Centralize matrices
    H = I(n) .- fill(1/n,n,n)
    cKxz = H * Kxz * H
    cKy = H * Ky * H
    cKz = H * Kz * H
    
    # Compute the test statistic
    Rz = test.ϵ .* pinv(cKz+test.ϵ.*I(n))
    Kxz_z = Rz * cKxz * Rz
    Ky_z = Rz * cKy * Rz
    test_stat = tr(Kxz_z*Ky_z) / n
    
    # Estimate the null distribution
    λ_Kxz_z = real.(eigvals(Kxz_z))
    λ_Kxz_z = sqrt.(λ_Kxz_z[λ_Kxz_z .>= test.min_λ])
    n_Kxz_z = length(λ_Kxz_z)
    v_Kxz_z = @views real.(eigvecs(Kxz_z)[:,((n-n_Kxz_z)+1):n]) .* λ_Kxz_z'

    λ_Ky_z = real.(eigvals(Ky_z))
    λ_Ky_z = sqrt.(λ_Ky_z[λ_Ky_z .>= test.min_λ])
    n_Ky_z = length(λ_Ky_z)
    v_Ky_z = @views real.(eigvecs(Ky_z)[:,((n-n_Ky_z)+1):n]) .* λ_Ky_z'
    
    v_prods = Matrix{Float64}(undef, n, n_Kxz_z*n_Ky_z)
    Threads.@threads for ij in 1:(n_Kxz_z*n_Ky_z)
        # Compute matrix indices from flattened matrix
        i, j = divrem(ij, n_Ky_z)
        j = j==0 ? n_Ky_z : j
        i = j==n_Ky_z ? i : i+1
        v_prods[:, ij] = @views v_Kxz_z[:,i] .* v_Ky_z[:,j]
    end
    if n > n_Kxz_z*n_Ky_z
        w = real.(eigvals(v_prods' * v_prods))
    else
        w = real.(eigvals(v_prods * v_prods'))
    end
    w = w[w .>= test.min_λ]
    
    # Sample from null distribution
    m = length(w)
    emp = Vector{Float64}(undef, test.B)
    Pchi = Chisq(1)
    @threads for i in 1:test.B
        emp[i] = dot(w,rand(Pchi,m)) / n
    end
    
    # Return the test statistic and the p-value
    return test_stat, mean(test_stat.<emp)
end


# Runs the test
function run_test(test::KCI, data::OneSampleData, α::Float64)
    test_stat, p = test_statistic(test, data.x, data.y, data.z)
    return TestSummary(test.name, test_stat, p<=α, pvalue=p)
end