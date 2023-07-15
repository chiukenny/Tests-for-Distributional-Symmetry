## Testing for rotation joint invariance about north geographic pole in SWARM data


# Seed for reproducibility
Random.seed!(1)


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
d = 4
α = 0.05


# Resampling parameters
B = 200
RS_G = Resampler(B=B, f_sampler=transform_sampler)
RS_sub = Resampler(B=B, f_sampler=subsampler)


# Group parameters
M = 2
GS_disc = GroupSampler(M, f_sample=rand_SWARM_disc_rot, f_transform=rotate_SWARM_rot)
GS_cont = GroupSampler(M, f_sample=rand_θ, f_transform=rotate_SWARM_rot)


# Data generation parameters
function SWARM_sample_data(n; train=false)
    # Subsample data
    data = train ? SWARM_tr : SWARM
    n_data = train ? n_SWARM_tr : n_SWARM
    inds = sample(1:n_data, n, replace=false)
    x = data[1:3, inds]
    y = reshape(data[4,inds], 1, n)
    σx = med_dist(x)
    σy = med_dist(y)
    xy = vcat(x, y)
    return OneSampleData(x=xy, σx=[σx,σy])
end
SWARM_tr_sample_data = n -> SWARM_sample_data(n, train=true)


# Output name
output_name = dir_out * "SWARM_joint_invariance_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")


# Run experiment
tests_disc = [
    Transform2S(GV_2S, MMD(GS=GS_disc,RS=RS_sub,f_kernel_mat=SWARM_Gaussian_kernel_mat))
    MMD(GV_MMD, GS=GS_disc, RS=RS_G, f_kernel_mat=SWARM_Gaussian_kernel_mat)
    NMMD(GV_NMMD, GS=GS_disc, RS=RS_G, J=ceil(sqrt(n)), f_kernel_mat=SWARM_Gaussian_kernel_mat)
    CW(GV_CW, GS=GS_disc, RS=RS_G, J=ceil(sqrt(n)))
]
tests_cont = [
    Transform2S(GV_2S, MMD(GS=GS_cont,RS=RS_sub,f_kernel_mat=SWARM_Gaussian_kernel_mat))
    MMD(GV_MMD, GS=GS_cont, RS=RS_G, f_kernel_mat=SWARM_Gaussian_kernel_mat)
    NMMD(GV_NMMD, GS=GS_cont, RS=RS_G, J=ceil(sqrt(n)), f_kernel_mat=SWARM_Gaussian_kernel_mat)
    CW(GV_CW, GS=GS_cont, RS=RS_G, J=ceil(sqrt(n)))
]
results = []

push!(results, run_tests(output_file, "jinv_disc", tests_disc, N=N, n=n, α=α,
                         f_sample_data=SWARM_sample_data, f_sample_tr_data=SWARM_tr_sample_data))
estimate_power(output_file, "jinv_disc", tests_disc, N=N, n=n, α=α,
               f_sample_data=SWARM_sample_data, f_sample_tr_data=SWARM_tr_sample_data)

push!(results, run_tests(output_file, "jinv_cont", tests_cont, N=N, n=n, α=α,
                         f_sample_data=SWARM_sample_data, f_sample_tr_data=SWARM_tr_sample_data))
estimate_power(output_file, "jinv_cont", tests_cont, N=N, n=n, α=α,
               f_sample_data=SWARM_sample_data, f_sample_tr_data=SWARM_tr_sample_data)

df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)