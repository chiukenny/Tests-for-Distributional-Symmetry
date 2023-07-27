## Testing for permutation equivariance in R^{10x10} data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
seed = 1
Random.seed!(seed)


# Experiment parameters
N = 1000
n = 100
d = 10
α = 0.05


# Data generation parameters
μ = zeros(d)
PW = MvNormal(μ, 1)

function sample_xy_permute(n; H0=true)
    W = rand(PW, d)
    Px = MvNormal(zeros(d), W*W')
    x = rand(Px, n)
    z = max_inv_permute(x)
    ty = zeros(d, n)
    @views begin
        Threads.@threads for i in 1:n
            Py = H0 ? MvNormal(x[:,i],1) : MvNormal(fill(x[1,i],d),1)
            τx = tau_inv_permute(x[:,i])
            y = rand(Py)
            ty[:,i] = permute(y, τx)
        end
    end
    return OneSampleData(x=x, y=ty, z=z)
end
H0_data = n -> sample_xy_permute(n, H0=true)
H1_data = n -> sample_xy_permute(n, H0=false)


# Test-specific parameters
B = 200
S = 50
N_tr = 100


# Output name
output_name = dir_out * "equivariance_exchangeability_d$(d)_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")


# Select kernel bandwidths
if train_kernel
    kci = KCI(GV_KCI, B=B)
    σxs = 10. .^ (-3:2)
    σys = 10. .^ (-5:2)
    σzs = 10. .^ (-3:2)
    kci_KS = optimize_kernel(kci, H0_data, f_sample_H1_data=H1_data, N=N_tr, n=n, α=α, σxs=σxs, σys=σys, σzs=σzs, seed=seed)
    
    cpt = CP(GV_CP, S=S, B=B, f_T=multiple_correlation_xyz)
    σys = 10. .^ (-6:1)
    σzs = 10. .^ (-2:2)
    cp_KS = optimize_kernel(cpt, H0_data, f_sample_H1_data=H1_data, N=N_tr, n=n, α=α, σys=σys, σzs=σzs, seed=seed)
    
    if save_kernel
        kci_obj = Dict(:test=>kci, :KS=>kci_KS)
        save_object(output_name*"_KCI.jld2", kci_obj)
        
        cp_obj = Dict(:test=>cpt, :KS=>cp_KS)
        save_object(output_name*"_CP.jld2", cp_obj)
    end
else
    kci = load_object(output_name*"_KCI.jld2")[:test]
    cpt = load_object(output_name*"_CP.jld2")[:test]
end


# Run experiment
tests = [
    kci
    cpt
]

results = []
push!(results, run_tests(output_file, "H0", tests, f_sample_data=H0_data, f_sample_tr_data=H0_data, N=N, n=n, α=α, seed=seed))
push!(results, run_tests(output_file, "H1", tests, f_sample_data=H1_data, f_sample_tr_data=H1_data, N=N, n=n, α=α, seed=seed))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)