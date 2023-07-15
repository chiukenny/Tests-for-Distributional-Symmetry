## Testing for invariance in LHC data


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
d = 4
α = 0.05


# Resampling parameters
B = 200
RS_G = Resampler(B=B, f_sampler=transform_sampler)
RS_sub = Resampler(B=B, f_sampler=subsampler)


# Group parameters
M = 2
GS = GroupSampler(M, f_sample=rand_θ1_θ2, f_transform=rotate_θ1_θ2)

rand_θ1_θ2_dep = () -> rand_θ1_θ2(paired=false)
GS_ind = GroupSampler(M, f_sample=rand_θ1_θ2_dep, f_transform=rotate_θ1_θ2)

rand_rotation_SO4 = () -> rand_rotation(4)
GS_SO4 = GroupSampler(M, f_sample=rand_rotation_SO4, f_transform=rotate_d)


# Data generation parameters
function LHC_sample_data(n::Int64; train=false)
    # Subsample data
    data = train ? LHC_tr : LHC
    n_data = train ? n_LHC_tr : n_LHC
    inds = sample(1:n_data, n, replace=false)
    x = @views data[[1,2,3,4], inds]
    σx = med_dist(x)
    return OneSampleData(x=x, σx=σx)
end
LHC_tr_sample_data = n -> LHC_sample_data(n, train=true)


# Output name
output_name = dir_out * "LHC_invariance_N$(N)_n$(n)_M$(M)_B$(B)"
output_file = open(output_name*".txt", "w")


# Run experiment
tests = [
    Transform2S(GV_2S, MMD(GS=GS,RS=RS_sub))
    MMD(GV_MMD, GS=GS, RS=RS_G)
    NMMD(GV_NMMD, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
    CW(GV_CW, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
]
tests_ind = [
    Transform2S(GV_2S, MMD(GS=GS_ind,RS=RS_sub))
    MMD(GV_MMD, GS=GS_ind, RS=RS_G)
    NMMD(GV_NMMD, GS=GS_ind, RS=RS_G, J=ceil(sqrt(n)))
    CW(GV_CW, GS=GS_ind, RS=RS_G, J=ceil(sqrt(n)))
]
tests_SO4 = [
    Transform2S(GV_2S, MMD(GS=GS_SO4,RS=RS_sub))
    MMD(GV_MMD, GS=GS_SO4, RS=RS_G)
    NMMD(GV_NMMD, GS=GS_SO4, RS=RS_G, J=ceil(sqrt(n)))
    CW(GV_CW, GS=GS_SO4, RS=RS_G, J=ceil(sqrt(n)))
]
results = []

push!(results, run_tests(output_file, "H0", tests, N=N, n=n, α=α,
                         f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data))

push!(results, run_tests(output_file, "H1_ind", tests_ind, N=N, n=n, α=α,
                         f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data))
estimate_power(output_file, "H1_ind", tests_ind, N=N, n=n, α=α,
               f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data)

push!(results, run_tests(output_file, "H1_SO4", tests_SO4, N=N, n=n, α=α,
                         f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data))
estimate_power(output_file, "H1_SO4", tests_SO4, N=N, n=n, α=α,
               f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data)

df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)