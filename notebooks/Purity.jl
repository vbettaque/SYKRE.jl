using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.SwapMatrix

N = 1
J = 1
q = 4
M = 2
L = 1000

G_init = inv(SwapMatrix.differential(M * L)) - 0.5 * I
G_init = BlockedArray(G_init, repeat([L], M), repeat([L], M))

weights_data = CSV.File("data/sre_weights/weights_M2_q4_L1000.csv") |> DataFrame

weights_purity = zeros(length(weights_data[:,1]))

for i = 1:length(weights_data[:,1])
    β = weights_data[i,1]
    w = weights_data[i,2]
    syk = SYKData(N, J, q, M, β)
    G, Σ = SwapMatrix.schwinger_dyson(G_init, w, syk; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)
    weights_purity[i] = SREMatrix.log2_saddle(G, Σ, syk) - 0.5
end

syk_data = CSV.File("data/syk_matrix/syk2_q4_L1000 copy.csv") |> DataFrame

p = plot(weights_data[:,1], weights_purity .- syk_data[:,2], label="difference")
p = plot!(weights_data[:,1], (weights_purity .- syk_data[:,2]) ./ syk_data[:,2], label="relative difference")
xlabel!("β")
ylabel!("log(Z)")



syk = SYKData(N, J, q, M, 10)
syk2 = SYKData(N, J, q, 1, 20)
G_w, Σ_w = SwapMatrix.schwinger_dyson(G_init, 0.33734130859375, syk; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)
G_syk, Σ_syk = SYKMatrix.schwinger_dyson(Matrix(G_init), syk2; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)

sum(abs.(G_syk - G_w)) / sum(abs.(G_syk))

# entropy = binary_entropy2(ws)
# entropy
# p = plot(data[:,1], data[:,2], label="q=4, L = 1000, β = 50")
# plot!(data[:,1], data[:,2] + entropy, label="saddle + entropy")

# xlabel!("\$ w \$")
# ylabel!("\$\\log(saddle)\$")

# display(p)
