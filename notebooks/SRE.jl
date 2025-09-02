using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.SwapMatrix
using .Replicas

N = 1
J = 1
q = 6
M = 4
L = 500

purity_data = CSV.File("data/sre_purities/purity_M4_q6_L500.csv") |> DataFrame

path = "data/sre/"
filename = "sre_M" * string(M) * "_q" * string(q) * "_L" * string(L) * ".csv"
file = path * filename
!ispath(path) && mkdir(path)
if !isfile(file)
    touch(file)
    write(file, "β,sre")
end

G_2_init = inv(SYKMatrix.differential(2 * L)) - I / 2
G_Z_init = inv(SYKMatrix.differential(L)) - I / 2

for i = 1:length(purity_data[:,1])
    β = purity_data[i,1]
    # β < 10 && continue
    purity_M = purity_data[i,2]
    syk_2 = SYKData(N, J, q, 1, 2*β)
    syk_Z = SYKData(N, J, q, 1, β)
    G_2, Σ_2 = SYKMatrix.schwinger_dyson(G_2_init, syk_2; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)
    purity_2 = SYKMatrix.log2_saddle(G_2, Σ_2, syk_2)
    println("purity_2 = ", purity_2)
    G_Z, Σ_Z = SYKMatrix.schwinger_dyson(G_Z_init, syk_Z; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)
    logZ = SYKMatrix.log2_saddle(G_Z, Σ_Z, syk_Z)
    println("logZ = ", logZ)
    sre = (purity_M - purity_2)/(1 - M÷2) + 2 * logZ

    df = DataFrame(β = β, sre = sre)
    CSV.write(file, df, append=true)
end
