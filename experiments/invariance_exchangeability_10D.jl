## Testing for permutation invariance in R^10 data


# Set the seed for reproducibility
Random.seed!(1)


# Experiment parameters
N = 1000
n = 200
d = 10
α = 0.05


# Resampling parameters
B = 200
RS_G = Resampler(B=B, f_sampler=transform_sampler)
RS_sub = Resampler(B=B, f_sampler=subsampler)


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


# Output name
output_name = dir_out * "invariance_exchangeability_d$(d)_N$(N)_n$(n)_M$(M)_B$(B)"
output_file = open(output_name*".txt", "w")


# Run experiment
tests = [
    Transform2S(GV_2S, MMD(GS=GS,RS=RS_sub))
    MMD(GV_MMD, GS=GS, RS=RS_G)
    NMMD(GV_NMMD, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
    CW(GV_CW, GS=GS, RS=RS_G, J=ceil(sqrt(n)))
    CW("CW-gen", multi_projection=false, GS=GS)
]

results = []
push!(results, run_tests(output_file, "H0+", tests, f_sample_data=H0_p_data, f_sample_tr_data=H0_p_data, N=N, n=n, α=α))
push!(results, run_tests(output_file, "H0-", tests, f_sample_data=H0_n_data, f_sample_tr_data=H0_n_data, N=N, n=n, α=α))
push!(results, run_tests(output_file, "H1", tests, f_sample_data=H1_r_data, f_sample_tr_data=H1_r_data, N=N, n=n, α=α))
estimate_power(output_file, "H1", tests, f_sample_data=H1_r_data, f_sample_tr_data=H1_r_data, N=N, n=n, α=α)
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)