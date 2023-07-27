## Helper functions for running experiments


using Distributions
using LinearAlgebra
using Random
using DataFrames
using Plots
using Base.Threads


# Runs invariance/equivariance tests and saves p-values
# Rejection rates and other outputs are logged in console
function run_tests(output_file::IOStream, exp_name::String, tests; N::Int64=1000, n::Int64=100, α::Float64=0.05,
                   f_sample_data::Function=f_nothing, f_sample_tr_data::Function=f_nothing, seed::Int64=randInt())
    # Prepare output data frame
    n_tests = length(tests)
    results = DataFrame()
    p_dict = Dict()
    t_dict = Dict()
    r_dict = Dict()
    names_len = 0
    for test in tests
        names_len = max(names_len, length(test.name))
        col_p = Symbol(exp_name * "_" * test.name * "_p")
        col_t = Symbol(exp_name * "_" * test.name * "_time")
        col_r = Symbol(exp_name * "_" * test.name * "_rej")
        p_dict[test.name] = col_p
        t_dict[test.name] = col_t
        r_dict[test.name] = col_r
        results[!,col_p] = Vector{Float64}(undef, N)
        results[!,col_r] = Vector{Bool}(undef, N)
        results[!,col_t] = Vector{Float64}(undef, N)
    end
    
    # Run independent simulations
    Random.seed!(seed)
    base_seed = randInt()
    for i in 1:N
        # Ensure the simulated data are consistent irrespective of threads at the minimum
        Random.seed!(base_seed + i)
        
        # Generate data
        data_tr = f_sample_tr_data(n)
        data = f_sample_data(n)
        
        for j in 1:n_tests
            # Initialize test
            test = deepcopy(tests[j])
            initialize(test, data_tr)
            
            # Run test and save results to data frame
            summary = @timed run_test(test, data, α)
            results[i,p_dict[test.name]] = summary.value.pvalue
            results[i,r_dict[test.name]] = summary.value.reject
            results[i,t_dict[test.name]] = summary.time
        end
    end
    
    # Print aggregated results
    write(output_file, "Experiment \"$(exp_name)\": [rej.rate] ± [rej.std] (avg.time)\n")
    for test in tests
        test_name = lpad(test.name, names_len, " ")
        rej_rate = round(mean(results[:,r_dict[test.name]]), digits=5)
        rej_std = round(sqrt(rej_rate*(1-rej_rate)/N), digits=5)
        avg_time = round(mean(results[:,t_dict[test.name]]), digits=5)
        write(output_file, "$(test_name): $(rpad(rej_rate,7,'0')) ± $(rpad(rej_std,7,'0')) ($(rpad(avg_time,7,'0'))s)\n")
    end
    write(output_file, "\n")
    
    return results
end


# Runs invariance/equivariance tests and saves aggregated results (no p-values)
function compare_tests(exp_name::String, tests; N::Int64=500, n::Int64=100, α::Float64=0.05,
                       f_sample_data::Function=f_nothing, f_sample_tr_data::Function=f_nothing, seed::Int64=randInt())
    # Prepare output data frame
    n_tests = length(tests)
    col_avgtime = Symbol(exp_name * "_avgtime")
    col_rej = Symbol(exp_name * "_rej")
    col_rej_sd = Symbol(exp_name * "_rej_sd")
    results = DataFrame(:Test => [test.name for test in tests],
                        col_avgtime => zeros(n_tests),
                        col_rej => zeros(n_tests),
                        col_rej_sd => zeros(n_tests))
    
    # Run independent simulations
    Random.seed!(seed)
    base_seed = randInt()
    for s in 1:N
        # Ensure the simulated data are consistent irrespective of threads at the minimum
        Random.seed!(base_seed + s)
        
        # Generate data
        data_tr = f_sample_tr_data(n)
        data = f_sample_data(n)
        
        for i in 1:n_tests
            # Initialize test
            test = deepcopy(tests[i])
            initialize(test, data_tr)
            
            # Run test and save results to data frame
            summary = @timed run_test(test, data, α)
            results[i,col_avgtime] += summary.time
            results[i,col_rej] += summary.value.reject
        end
    end
    
    # Clean up results
    results[:,col_avgtime] = results[:,col_avgtime] ./ N
    results[:,col_rej] = results[:,col_rej] ./ N
    results[:,col_rej_sd] = vec(sqrt.(results[:,col_rej].*(1 .- results[:,col_rej])./N))
    return results
end


# Estimates the test power based on resampling
function estimate_power(output_file::IOStream, exp_name::String, tests; N::Int64=500, n::Int64=100, α::Float64=0.05,
                        f_sample_data::Function=f_nothing, f_sample_tr_data::Function=f_nothing, seed::Int64=randInt())
    n_tests = length(tests)
    
    # Ensure the simulated data are consistent irrespective of threads
    Random.seed!(seed)
    base_seed = randInt()
    Random.seed!(base_seed)
    
    # Generate data
    data_tr = f_sample_tr_data(n)
    data = f_sample_data(n)
    
    # Initialize tests
    names_len = 0
    for i in 1:n_tests
        test = tests[i]
        names_len = max(names_len, length(test.name))
        initialize(test, data_tr)
    end
    
    pows = zeros(n_tests, N)
    @threads for j in 1:N
        # Ensure the simulated data are consistent irrespective of threads
        Random.seed!(base_seed + j)
        
        # Resample data
        rs_x = data.x[:, sample(1:n,n,replace=true)]
        rs_data = OneSampleData(x=rs_x)
        
        for i in 1:n_tests
            # Run test
            test = tests[i]
            B = hasfield(typeof(test),:RS) ? test.RS.B : test.test.RS.B
            summary = run_test(test, rs_data, α)
            
            # Estimate power
            p0 = max(0, (summary.pvalue*(B+1)-1)/B)  # May have some small negative numbers due to numerical issues
            P_bin = Binomial(B, p0)
            pows[i,j] = cdf(P_bin, floor(Int64,α*(B+1)-1))
        end
    end
    
    # Print aggregated results
    write(output_file, "Power estimates for \"$(exp_name)\": [pow.est] ± [pow.std]\n")
    for i in 1:n_tests
        test = tests[i]
        test_name = lpad(test.name, names_len, " ")
        pow_est = round(mean(pows[i,:]), digits=5)
        pow_std = round(sqrt(pow_est*(1-pow_est)/N), digits=5)
        write(output_file, "$(test_name): $(rpad(pow_est,7,'0')) ± $(rpad(pow_std,7,'0'))\n")
    end
    write(output_file, "\n")
end