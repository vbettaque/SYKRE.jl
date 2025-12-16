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

weights_data = CSV.File("data/sre_weights/weights_M4_q6_L500.csv") |> DataFrame

path = "data/sre_purities/"
filename = "purity_M" * string(M) * "_q" * string(q) * "_L" * string(L) * ".csv"
file = path * filename
!ispath(path) && mkdir(path)
if !isfile(file)
    touch(file)
    write(file, "β,purity,weight")
end

G_init = Replicas.init(M, L)

purities = zeros(length(weights_data[:,1]))

for i = 1:length(purities)
    β = weights_data[i,1]
    β < 29 && continue
    w = weights_data[i,2]
    syk = SYKData(N, J, q, M, β)
    tol = max((β * J / L)^2 / 10, 1e-5)
    G, Σ = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = 0.005, lerp_divisor = 2, tol=tol, max_iters=5000)
    purity = PurityReplicas.log2_saddle(G, Σ, syk)

    df = DataFrame(β = β, purity = purity, weight = w)
    CSV.write(file, df, append=true)
end
