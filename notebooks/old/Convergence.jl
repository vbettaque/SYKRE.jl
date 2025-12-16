using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier

function generate_matrix_syk_data(L, M, q, βs)
    @assert M > 0
    N = 1
    J = 1

    path = "data/syk_matrix/"
    filename = "syk" * string(M) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "β,logZ\n")
    end

    G_0 = -inv(SYKMatrix.differential(M * L))'
    G_init = G_0

    for i in eachindex(βs)
        β = βs[i]
        println(i, " out of ", length(βs), ": β = ", β)

        syk = SYKData(N, J, q, 1, M * β)
        t = isone(i) ? 0.5 : 0.5

        G, Σ = SYKMatrix.schwinger_dyson(G_init, syk; init_lerp = t, lerp_divisor = 2, max_iters=1000)

        logZ = SYKMatrix.log2Z_saddle(G, Σ, syk)

        df = DataFrame(β = β, logZ = logZ)
        CSV.write(file, df, append=true)

        # G_init = G
    end

end

βs = LinRange(0.1, 10, 100)
generate_matrix_syk_data(1000, 2, 4, βs)

data_consec = CSV.File("data/syk_matrix/syk1_q4_L1000_consec.csv"; select=["β", "logZ"]) |> DataFrame
data_fixed = CSV.File("data/syk_matrix/syk1_q4_L1000_fixed.csv"; select=["β", "logZ"]) |> DataFrame

p = plot(data_consec[:, 1], data_consec[:, 2], label="consecutive initial values")
p = plot!(data_fixed[:, 1], data_fixed[:, 2], label="fixed initial values")

xlabel!("\$ β \$")
ylabel!("\$\\log(Z)\$")
title!("SYK \$(q=4, M=1, J=1, \\Delta\\tau = 0.1)\$")

display(p)

data_consec = CSV.File("data/syk_matrix/syk2_q4_L1000_consec.csv"; select=["β", "logZ"]) |> DataFrame
data_fixed = CSV.File("data/syk_matrix/syk2_q4_L1000_fixed.csv"; select=["β", "logZ"]) |> DataFrame

p = plot(data_consec[:, 1], data_consec[:, 2], label="consecutive initial values")
p = plot!(data_fixed[:, 1], data_fixed[:, 2], label="fixed initial values")

xlabel!("\$ β \$")
ylabel!("\$\\log(Z)\$")
title!("SYK \$(q=4, M=2, J=1, \\Delta\\tau = 0.1)\$")

display(p)