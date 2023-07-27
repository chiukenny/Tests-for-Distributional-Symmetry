## Investigating the effect of number of random projections on NMMD and CW
## in tests for permutation invariance in R^10 data


# Set the seed to ensure that simulated data are at least consistent irrespective of threads
seed = 1
Random.seed!(seed)


# Experiment parameters
N = 1000
n = 200
d = 10
α = 0.05


# Resampling parameters
B = 200
RS = Resampler(B=B, f_sampler=transform_sampler)


# Group parameters
M = 2
rand_permutation_d = () -> randperm(d)
permute_generators_d = () -> get_permute_generators(d)
GS = GroupSampler(M, f_sample=rand_permutation_d, f_transform=permute, f_get_generators=permute_generators_d)


# Data generation
μ = zeros(d)

Σ0_p = diagm(ones(d).*(1-1/d)) .+ ones(d,d)./d
P0_p = MvNormal(μ, Σ0_p)
H0_p_data = n -> sample_data(n, P0_p)

Σ0_n = diagm(ones(d).*(1+1/(d-1))) .- ones(d,d)./(d-1)
P0_n = MvNormal(μ, Σ0_n)
H0_n_data = n -> sample_data(n, P0_n)

PW = MvNormal(μ, 1)
H1_r_data = n -> begin
    W = rand(PW, d)
    P1 = MvNormal(μ, W*W')
    return sample_data(n, P1)
end

# Run experiment
results = []
Js = collect(5:5:50)
for i in 1:length(Js)
    tests = [
        NMMD(GV_NMMD, GS=GS, RS=RS, J=Js[i])
        CW(GV_CW, GS=GS, RS=RS, J=Js[i])
    ]
    push!(results, compare_tests("H0+_J$(Js[i])", tests, f_sample_data=H0_p_data, f_sample_tr_data=H0_p_data, N=N, n=n, α=α, seed=seed))
    push!(results, compare_tests("H0-_J$(Js[i])", tests, f_sample_data=H0_n_data, f_sample_tr_data=H0_n_data, N=N, n=n, α=α, seed=seed))
    push!(results, compare_tests("H1_J$(Js[i])", tests, f_sample_data=H1_r_data, f_sample_tr_data=H1_r_data, N=N, n=n, α=α, seed=seed))
end
df = innerjoin(on=:Test, results...)
CSV.write(dir_out*"invariance_exchangeability_d$(d)_N$(N)_n$(n)_M$(M)_B$(B)_Jvar.csv", df)