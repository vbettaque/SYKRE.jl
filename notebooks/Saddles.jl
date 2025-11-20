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

R = 2

syk = SYKData(N, J, q, R, β)

powervec(ss, n) = Iterators.product(ntuple(i->ss, n)...)

for p in powervec([0, 1, -1], R÷2)
    # all(iszero.(p)) && continue
    G_init = Replicas.init(R, L)
    # if !isnothing(findfirst(p .< 0))
    #     isnothing(findfirst(p .> 0)) && continue
    #     findfirst(p .< 0) < findfirst(p .> 0) && continue
    # end
    if iseven(R)
        for i = 1:(R÷2-1)
            @views G_init.blocks[:, :, i+1] .*= p[i]
            @views G_init.blocks[:, :, R-i+1] .*= p[i]
        end
        @views G_init.blocks[:, :, R÷2+1] .*= p[R÷2]
    else
        for i = 1:((R-1)÷2)
            @views G_init.blocks[:, :, i+1] .*= p[i]
            @views G_init.blocks[:, :, R-i+1] .*= p[i]
        end
    end
    WeightReplicas.plot_matrix(G_init; title="$(p)")
    G, Σ = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = 0.1, lerp_divisor = 2, tol=1e-10, max_iters=1000)
    log2_saddle = WeightReplicas.log2_saddle(G, Σ, w, syk)
    WeightReplicas.plot_matrix(G; title="log2_saddle = $(log2_saddle)")
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
