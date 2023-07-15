## Investigating the effect of number of random projections on NMMD and CW
## in tests for invariance in LHC data


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
RS = Resampler(B=B, f_sampler=subsample_transformer)


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


# Run experiments
Js = collect(2:2:20)
results = []
for i in 1:length(Js)
    tests = [
        NMMD(GV_NMMD, GS=GS, RS=RS, J=Js[i])
        CW(GV_CW, GS=GS, RS=RS, J=Js[i])
    ]
    tests_ind = [
        NMMD(GV_NMMD, GS=GS_ind, RS=RS, J=Js[i])
        CW(GV_CW, GS=GS_ind, RS=RS, J=Js[i])
    ]
    tests_SO4 = [
        NMMD(GV_NMMD, GS=GS_SO4, RS=RS, J=Js[i])
        CW(GV_CW, GS=GS_SO4, RS=RS, J=Js[i])
    ]
    push!(results, compare_tests("H0_J$(Js[i])", tests, N=N, n=n, α=α,
                                 f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data))
    push!(results, compare_tests("H1_ind_J$(Js[i])", tests_ind, N=N, n=n, α=α,
                                 f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data))
    push!(results, compare_tests("H1_SO4_J$(Js[i])", tests_SO4, N=N, n=n, α=α,
                                 f_sample_data=LHC_sample_data, f_sample_tr_data=LHC_tr_sample_data))
end
df = innerjoin(on=:Test, results...)
CSV.write(dir_out*"LHC_invariance_N$(N)_n$(n)_M$(M)_B$(B)_Jvar.csv", df)