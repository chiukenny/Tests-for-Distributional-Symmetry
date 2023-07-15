## ## Testing for conditional Lorentz invariance in top quark data


# Set the seed for reproducibility
Random.seed!(1)


# Read in data
fid = h5open(dir_out_dat*"TQT.h5", "r")
TQT = read(fid["data"])
TQT_tr = read(fid["train"])
close(fid)
n_TQT = size(TQT, 2)
n_TQT_tr = size(TQT_tr, 2)


# Experiment parameters
N = 1000
n = 100
α = 0.05


# Group parameters
dz = 2
tau_inv_lorentz_dz = x -> tau_inv_lorentz(x, dz=dz)


# Data generation parameters
function TQT_sample_data(n::Int64; train=false, H1=false)
    # Subsample data
    data = train ? TQT_tr : TQT
    n_data = train ? n_TQT_tr : n_TQT
    inds = sample(1:n_data, n, replace=false)
    @views begin
        x = data[1:(4*dz), inds]
        y = H1 ? zeros(1, n) : reshape(data[(4*dz)+1,inds], 1, n)
        z = zeros(dz, n)
        Threads.@threads for i in 1:n
            if H1
                y[i] = rand(Bernoulli(0.9)) ? x[1,i]>=200 : x[1,i]<200
            end
            z[:,i] = tau_inv_lorentz_dz(x[:,i])
        end
    end
    σx = med_dist(x)
    σz = med_dist(z)
    return OneSampleData(x=x, y=y, z=z, σx=σx, σz=σz)
end
TQT_tr_sample_H0_data = n -> TQT_sample_data(n, train=true, H1=false)
TQT_tr_sample_H1_data = n -> TQT_sample_data(n, train=true, H1=true)
TQT_sample_H1_data = n -> TQT_sample_data(n, train=false, H1=true)


# Test-specific functions
Ky = binary_kernel_mat
k_params = Dict(:y => Float64[])


# Test-specific parameters
B = 200
N_tr = 100


# Output name
output_name = dir_out * "TQT_conditional_invariance_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")


# Select kernel bandwidths
if train_kernel
    kci = KCI(GV_KCI, B=B, f_kernel_mat_y=Ky, k_params=k_params)
    σxs = collect(LinRange(5, 50, 19))
    σzs = collect(LinRange(5, 100, 39))
    @time kci_KS = optimize_kernel(kci, TQT_tr_sample_H0_data, f_sample_H1_data=TQT_tr_sample_H1_data,
                                   N=N_tr, n=n, α=α, σxs=σxs, σzs=σzs)
    
    if save_kernel
        save_test(output_name*"_KCI.jld2", kci, kci_KS)
    end
else
    kci = load_test(output_name*"_KCI.jld2")[:test]
end


# Run experiment
tests = [
    kci
]

results = []
push!(results, run_tests(output_file, "H0", tests, N=N, n=n, α=α,
                         f_sample_data=TQT_sample_data, f_sample_tr_data=TQT_tr_sample_H0_data))
push!(results, run_tests(output_file, "H1", tests, N=N, n=n, α=α,
                         f_sample_data=TQT_sample_H1_data, f_sample_tr_data=TQT_tr_sample_H1_data))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)