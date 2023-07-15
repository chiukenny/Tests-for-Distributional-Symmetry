## Testing for equivariance in LHC data


# Set the seed for reproducibility
Random.seed!(1)


# Read in data
fid = h5open(dir_out_dat*"LHC.h5", "r")
LHC = copy(read(fid["data"]))
LHC_tr = copy(read(fid["train"]))
close(fid)
n_LHC = size(LHC, 2)
n_LHC_tr = size(LHC_tr, 2)


# Experiment parameters
N = 1000
n = 100
α = 0.05


# Data generation
GS = GroupSampler(f_sample=rand_θ1_θ2, f_transform=rotate_θ1_θ2)
function LHC_sample_data(n::Int64; train=false, H1=false)
    @views begin
        # Subsample data
        data = train ? LHC_tr : LHC
        n_data = train ? n_LHC_tr : n_LHC
        inds = sample(1:n_data, n, replace=false)
        xy = train && !H1 ? transform_all(GS,data[:,inds]) : data[:,inds]
        x = xy[1:2, :]
        y = xy[3:4, :]
        
        # Normalize data
        σ2 = std(hcat(x,y), dims=2)
        x = x ./ σ2
        y = y ./ σ2
        
        z = max_inv_rotate(x)
        if H1
            return OneSampleData(x=x, y=y, z=z)
        end
        ty = zeros(2, n)
        Threads.@threads for i in 1:n
            τ_x = tau_inv_rotate_2D(x[:,i])
            ty[:,i] = rotate_2D(y[:,i], τ_x)
        end
    end
    return OneSampleData(x=x, y=ty, z=z)
end
LHC_tr_sample_data = n -> LHC_sample_data(n, train=true)
LHC_tr_sample_H1_data = n -> LHC_sample_data(n, train=true, H1=true)
LHC_sample_H1_data = n -> LHC_sample_data(n, H1=true)


# Test-specific parameters
B = 200
S = 50
N_tr = 50


# Output name
output_name = dir_out * "LHC_equivariance_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")


# Select kernel bandwidths
if train_kernel
    kci = KCI(GV_KCI, B=B)
    σxs = 10. .^ (-1:1)
    σys = 10. .^ (-2:1)
    σzs = 10. .^ (-1:1)
    kci_KS = optimize_kernel(kci, LHC_tr_sample_data, f_sample_H1_data=LHC_tr_sample_H1_data,
                             N=N_tr, n=n, α=α, σxs=σxs, σys=σys, σzs=σzs)
    
    cpt = CP(GV_CP, S=S, B=B, f_T=multiple_correlation_xyz)
    σys = 10. .^ (-3:-1)
    σzs = 10. .^ (-3:-1)
    cp_KS = optimize_kernel(cpt, LHC_tr_sample_data, f_sample_H1_data=LHC_tr_sample_H1_data,
                            N=N_tr, n=n, α=α, σys=σys, σzs=σzs)
    if save_kernel
        save_test(output_name*"_KCI.jld2", kci, kci_KS)
        save_test(output_name*"_CP.jld2", cpt, cp_KS)
    end
else
    kci = load_test(output_name*"_KCI.jld2")[:test]
    cpt = load_test(output_name*"_CP.jld2")[:test]
end


# Run experiment
tests = [
    kci
    cpt
]

results = []
push!(results, run_tests(output_file, "H0", tests, N=N, n=n, α=α,
                         f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data))
push!(results, run_tests(output_file, "H1", tests, N=N, n=n, α=α,
                         f_sample_data=LHC_sample_H1_data, f_sample_tr_data=LHC_tr_sample_H1_data))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)