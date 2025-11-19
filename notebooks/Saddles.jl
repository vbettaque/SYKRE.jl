using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays, Combinatorics

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.Replicas
using SYKRE.WeightReplicas
using SYKRE.PurityReplicas
using SYKRE.SwapMatrix

N = 1
J = 1
q = 4
M = 4
β = 5
L = 500
w = 0.5

R = 3

syk = SYKData(N, J, q, R, β)

powervec(ss, n) = Iterators.product(ntuple(i->ss, n)...)

for p in powervec([0, 1, -1], R÷2)
    G_init = Replicas.init(R, L)
    leading_zero = true
    findfirst(p .< 0) < findfirst(p .> 0) && continue
    for i = 1:(R÷2-1)
        @views G_init.blocks[:, :, i+1] .*= p[i]
        @views G_init.blocks[:, :, R-i+1] .*= p[i]
    end
    @views G_init.blocks[:, :, R÷2+1] .*= p[R÷2]
    WeightReplicas.plot_matrix(G_init; title="$(p)")
    WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = 0.5, lerp_divisor = 2, tol=1e-10, max_iters=1000)
end

# G_init_1 = Replicas.init(1, 2*L)
# G_init_2 = Replicas.init(2, L)

# G_init_4 = Replicas.init(4, L)
# @views G_init_4.blocks[:, :, 2] .= 0
# @views G_init_4.blocks[:, :, 4] .= 0

# syk_4 = SYKData(N, J, q, 4, β)

# WeightReplicas.schwinger_dyson(G_init_4, w, syk_4; init_lerp = 0.1, lerp_divisor =2, tol=1e-10, max_iters=1000)


# G_init_6 = Replicas.init(6, L)
# @views G_init_6.blocks[:, :, 2] .= 0
# @views G_init_6.blocks[:, :, 4] .= 0
# @views G_init_6.blocks[:, :, 6] .= 0
# WeightReplicas.plot_matrix(G_init_6)

# syk_6 = SYKData(N, J, q, 6, β)

# WeightReplicas.schwinger_dyson(G_init_6, w, syk_6; init_lerp = 0.1, lerp_divisor =2, tol=1e-10, max_iters=1000)