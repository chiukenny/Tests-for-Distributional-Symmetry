## Testing for conditional invariance wrt 2D rotations about axes in SWARM data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
seed = 1
Random.seed!(seed)


# Read in data
fid = h5open(dir_out_dat*"SWARM.h5", "r")
SWARM = copy(read(fid["data"]))
SWARM_tr = copy(read(fid["train"]))
close(fid)
n_SWARM = size(SWARM, 2)
n_SWARM_tr = size(SWARM_tr, 2)


# Experiment parameters
N = 1000
n = 220
α = 0.05


# Data generation parameters
function SWARM_sample_data(n, GS; train=false, H0=false)
    # Subsample data
    data = train ? SWARM_tr : SWARM
    n_data = train ? n_SWARM_tr : n_SWARM
    inds = sample(1:n_data, n, replace=false)
    @views begin
        x = train && H0 ? transform_all(GS,data[1:3,inds]) : data[1:3,inds]
        y = reshape(data[4,inds], 1, n)
    end
    z = GS.f_max_inv(x)
    σx = med_dist(x)
    σy = med_dist(y)
    σz = med_dist(z)
    return OneSampleData(x=x, y=y, z=z, σx=σx, σy=σy, σz=σz)
end
GS3 = GroupSampler(f_sample=()->rand_3D_axis(ax=3), f_transform=rotate_d, f_max_inv=x->max_inv_rotate_axis(x,ax=3))
SWARM_tr_sample_data = n -> SWARM_sample_data(n, GS3, train=true, H0=true)

GS1 = GroupSampler(f_sample=rand_3D_axis, f_transform=rotate_d, f_max_inv=x->max_inv_rotate_axis(x,ax=1))
SWARM_tr_sample_H1_data = n -> SWARM_sample_data(n, GS1, train=true, H0=false)


# Test-specific parameters
B = 200
N_tr = 100


# Output name
output_name = dir_out * "SWARM_conditional_invariance_N$(N)_n$(n)_B$(B)"


# Select kernel bandwidths
if train_kernel
    kci = KCI(GV_KCI, B=B)
    σxs = collect(LinRange(1, 3, 13))
    σys = collect(LinRange(5, 15, 31))
    σzs = collect(LinRange(5e-3, 6e-3, 3))
    kci_KS = optimize_kernel(kci, SWARM_tr_sample_data, f_sample_H1_data=SWARM_tr_sample_H1_data,
                             N=N_tr, n=n, α=α, σxs=σxs, σys=σys, σzs=σzs, seed=seed)
    
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
output_file = open(output_name*".txt", "w")

# Rotations about axes
for (sym,d) in [("X_inv",1), ("Y_inv",2), ("Z_inv",3)]
    GS = GroupSampler(f_max_inv = x->max_inv_rotate_axis(x,ax=d))
    SWARM_tr_sample_data_d = n -> SWARM_sample_data(n, GS, train=true)
    SWARM_sample_data_d = n -> SWARM_sample_data(n, GS)
    push!(results, run_tests(output_file, sym, tests, N=N, n=n, α=α, 
                             f_sample_data=SWARM_sample_data_d, f_sample_tr_data=SWARM_tr_sample_data_d, seed=seed))
end

# Rotations about north geomagnetic pole
GS_NP = GroupSampler(f_max_inv = max_inv_geoNP)
SWARM_tr_sample_data_NP = n -> SWARM_sample_data(n, GS_NP, train=true)
SWARM_sample_data_NP = n -> SWARM_sample_data(n, GS_NP, train=false)
push!(results, run_tests(output_file, "NP", tests, N=N, n=n, α=α,
                         f_sample_data=SWARM_sample_data_NP, f_sample_tr_data=SWARM_tr_sample_data_NP, seed=seed))

results_df = hcat(results...)
CSV.write(output_name*".csv", results_df)
close(output_file)