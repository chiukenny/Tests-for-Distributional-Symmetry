## Investigating the effect of number of sampled group actions on tests for SO(4) invariance in R^4 data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
seed = 1
Random.seed!(seed)


# Experiment parameters
N = 1000
n = 200
d = 4
α = 0.05


# Resampling parameters
B = 200
RS_G = Resampler(B=B, f_sampler=transform_sampler)
RS_sub = Resampler(B=B, f_sampler=subsampler)


# Group parameters
Ms = collect(1:5)
rand_rotation_d = () -> rand_rotation(d)


# Data generation parameters
σ = 1

μ0 = zeros(d)
P0 = MvNormal(μ0, σ)
H0_data = n -> sample_data(n, P0)

μ1 = zeros(d)
μ1[1] = 0.4
P1 = MvNormal(μ1, σ)
H1_data = n -> sample_data(n, P1)


# Run experiment
results = []
for i in 1:length(Ms)
    GS = GroupSampler(Ms[i], f_sample=rand_rotation_d, f_transform=rotate_d)
    
    tests = [
        MMD(GV_MMD, GS=GS, RS=RS_G)
        NMMD(GV_NMMD, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
        CW(GV_CW, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
    ]
    push!(results, compare_tests("H0_M$(Ms[i])", tests, f_sample_data=H0_data, f_sample_tr_data=H0_data, N=N, n=n, α=α, seed=seed))
    push!(results, compare_tests("H1_M$(Ms[i])", tests, f_sample_data=H1_data, f_sample_tr_data=H1_data, N=N, n=n, α=α, seed=seed))
end
df = innerjoin(on=:Test, results...)
CSV.write(dir_out*"invariance_rotation_d$(d)_N$(N)_n$(n)_Mvar_B$(B).csv", df)