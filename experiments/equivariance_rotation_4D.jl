## Testing for SO(4) equivariance in R^{4x4} data


# Set the seed for reproducibility
Random.seed!(1)


# Experiment parameters
N = 1000
n = 50
d = 4
α = 0.05


# Data generation
μ = zeros(d)
PW = MvNormal(μ, 1)

function sample_xy_rotate(n; H0=true)
    W = rand(PW, d)
    Σ = W * W'
    Px = MvNormal(μ, Σ)
    x = rand(Px, n)
    z = max_inv_rotate(x)
    ty = zeros(d, n)
    @views begin
        Threads.@threads for i in 1:n
            tau_x = tau_inv_rotate(x[:,i])
            Py = H0 ? MvNormal(x[:,i],1) : MvNormal(abs.(x[:,i]),1)
            y = rand(Py)
            ty[:,i] = rotate_d(y, tau_x)
        end
    end
    σx = med_dist(x)
    σy = med_dist(ty)
    σz = med_dist(z)
    return OneSampleData(x=x, y=ty, z=z, σx=σx, σy=σy, σz=σz)
end
H0_data = n -> sample_xy_rotate(n, H0=true)
H1_data = n -> sample_xy_rotate(n, H0=false)


# Test-specific parameters
B = 200


# Output name
output_name = dir_out * "equivariance_rotation_d$(d)_N$(N)_n$(n)_B$(B)"
output_file = open(output_name*".txt", "w")


# Run experiment
tests = [
    KCI(GV_KCI, B=B)
    CP(GV_CP, S=50, B=B, f_T=multiple_correlation_xyz)
]

results = []
push!(results, run_tests(output_file, "H0", tests, f_sample_data=H0_data, f_sample_tr_data=H0_data, N=N, n=n, α=α))
push!(results, run_tests(output_file, "H1", tests, f_sample_data=H1_data, f_sample_tr_data=H1_data, N=N, n=n, α=α))
df = hcat(results...)
CSV.write(output_name*".csv", df)
close(output_file)