using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays, Combinatorics, BenchmarkTools, Profile

using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas

function generate_Z_data(βs, q, L; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    G_init = Replicas.init(1, L)

    G_temp = Replicas.init(1, L)
    Σ_temp = Replicas.init(1, L)

    path = "data/partition_function/"
    filename = "partition_function" * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "beta,Z,pf_minus,pf_plus,on_shell_term\n")
    end

    for i in eachindex(βs)
        β = βs[i]
        println(i, " out of ", length(βs), ": β = ", β)

        Z = 0
        pf_minus = 0
        pf_plus = 0
        on_shell_term = 0

        syk = SYKData(1, 1, q, 1, β)
        G_temp, Σ_temp = WeightedReplicas.schwinger_dyson(G_init, 0, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)

        Z = WeightedReplicas.log_saddle(G_temp, Σ_temp, 0, syk)
        pf_minus, pf_plus = WeightedReplicas.pfaffians(Σ_temp, syk)
        on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * (syk.β / L)^2 * sum(x -> x^syk.q, G_temp)

        df = DataFrame(beta = β, Z = Z, pf_minus = pf_minus, pf_plus = pf_plus, on_shell_term=on_shell_term)

        CSV.write(file, df, append=true)
    end

end

L = 1000
βs = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

generate_Z_data(βs, 2, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
generate_Z_data(βs, 4, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
generate_Z_data(βs, 6, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
generate_Z_data(βs, 8, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
