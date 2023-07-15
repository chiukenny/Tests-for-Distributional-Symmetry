## Testing for SO(3)-invariance in R^3 data via representative inversions


# Set the seed for reproducibility
Random.seed!(1)


# Experiment parameters
N = 1000
n = 100
α = 0.05


# Resampling parameters
B = 200
RS_G = Resampler(B=B, f_sampler=transform_sampler)
RS_sub = Resampler(B=B, f_sampler=subsampler)

# Samples directly from Haar measure over SO(3)
param_sampler = (test,x) -> begin
    n = size(x, 2)
    bx = zeros(4, n)
    Threads.@threads for i in 1:n
        bx[:,i] = rand_3D(quat=true)
    end
    return bx
end
RS_par = Resampler(B=B, f_sampler=param_sampler)


# Group parameters
M = 2
rand_3D_q = () -> rand_3D(quat=true)
rotate_3D_q = (x,q) -> rotate_3D(x,q,quat=true)
GS = GroupSampler(M, f_sample=rand_3D_q, f_transform=rotate_3D_q)


# Data generation parameters
function SO3_data(n, P)
    y = rand(P, n)
    x = zeros(4, n)
    Threads.@threads for i in 1:n
        x[:,i] = @views mat_to_quat(tau_inv_rotate(y[:,i]))
    end
    return OneSampleData(x=x)
end

μ = zeros(3)

P0 = MvNormal(μ, 1)
H0_data = n -> SO3_data(n, P0)

PW = MvNormal(zeros(3), 1)
H1_data = n -> begin
    W = rand(PW, 3)
    return SO3_data(n, MvNormal(μ,W*W'))
end


# Output name
output_name = dir_out * "invariance_SO3_N$(N)_n$(n)_M$(M)_B$(B)"
output_file = open(output_name*".txt", "w")


# Run experiment
Kx = SO3_kernel_mat
tests = [
    Transform2S(GV_2S, MMD(GS=GS,RS=RS_sub,f_kernel_mat=Kx))
    MMD(GV_MMD, GS=GS, RS=RS_G, f_kernel_mat=Kx)
    MMD("MMD-par", GS=GS, RS=RS_par, f_kernel_mat=Kx)
    NMMD(GV_NMMD, GS=GS, RS=RS_G, J=ceil(sqrt(n)), f_kernel_mat=Kx)
]

results = []
push!(results, run_tests(output_file, "H0", tests, f_sample_data=H0_data, f_sample_tr_data=H0_data, N=N, n=n, α=α))
push!(results, run_tests(output_file, "H1", tests, f_sample_data=H1_data, f_sample_tr_data=H1_data, N=N, n=n, α=α))
estimate_power(output_file, "H1", tests, f_sample_data=H1_data, f_sample_tr_data=H1_data, N=N, n=n, α=α)
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)