using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays, Combinatorics

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.Replicas
using SYKRE.WeightedReplicas
using SYKRE.PurityReplicas
using SYKRE.WeightedMatrix

N = 1
J = 1
q = 2
β = 50
L = 1000
w = 0.5

R = 4

syk = SYKData(N, J, q, R, β)

powervec(ss, n) = Iterators.product(ntuple(i->ss, n)...)

# G_single_init = Replicas.init(1, R * L)
# G_single, _ = WeightReplicas.schwinger_dyson(G_single_init, 0, SYKData(N, J, q, 1, R * β); init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)

G_init = Replicas.init(R, L)

# for i = 1:R
#     row_range = ((i-1) * L + 1):(i * L)
#     @views G_init.blocks[:, :, i] = G_single.blocks[row_range, 1:L, 1]
# end

WeightedReplicas.plot_matrix(G_init; title="")


for p in powervec([0, 1, -1], R÷2)
    all(iszero.(p)) && continue
    G_p = Replicas.init(R, L)
    @views G_p.blocks[:, :, :] = copy(G_init.blocks[:, :, :])
    if !isnothing(findfirst(p .< 0))
        isnothing(findfirst(p .> 0)) && continue
        findfirst(p .< 0) < findfirst(p .> 0) && continue
    end
    if iseven(R)
        for i = 1:(R÷2-1)
            @views G_p.blocks[:, :, i+1] .*= p[i]
            @views G_p.blocks[:, :, R-i+1] .*= p[i]
        end
        @views G_p.blocks[:, :, R÷2+1] .*= p[R÷2]
    else
        for i = 1:((R-1)÷2)
            @views G_p.blocks[:, :, i+1] .*= p[i]
            @views G_p.blocks[:, :, R-i+1] .*= p[i]
        end
    end
    WeightReplicas.plot_matrix(G_p; title="$(p)")
    G, Σ = WeightReplicas.schwinger_dyson(G_p, w, syk; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
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


# N = 1
# J = 1
# q = 4
# M = 4
# β = 2
# L = 500
# w = 0.5

# R = 4

# syk = SYKData(N, J, q, R, β)

# # Permutation 1

# G_init = inv(SwapMatrix.differential(R * L; M = 1, periodic=false)./2) - I
# G_init = BlockedArray(G_init, repeat([L], R), repeat([L], R))

# G_init[Block(1, 2)] .= 0
# G_init[Block(2, 1)] .= 0

# G_init[Block(1, 4)] .= 0
# G_init[Block(4, 1)] .= 0

# G_init[Block(2, 3)] .= 0
# G_init[Block(3, 2)] .= 0

# G_init[Block(3, 4)] .= 0
# G_init[Block(4, 3)] .= 0

# G, Σ = SwapMatrix.schwinger_dyson(G_init, w, syk; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)

# blue = RGB(0,101/255,1)
# orange = RGB(1,154/255,0)
# grad = cgrad([blue, :gray95, orange], [0.0, 0.5, 1.0])
# p = heatmap(G, aspect_ratio = 1, clims=(-1, 1), yflip = true, color = grad, xaxis=false, yaxis=false, colorbar=false, grid=false, top_margin = -10Plots.cm, bottom_margin = -10Plots.cm, left_margin = -10Plots.cm, right_margin = -10Plots.cm, size=(512,512))

# # Permutation 2

# G_init = inv(SwapMatrix.differential(R * L; M = 1, periodic=false)./2) - I
# G_init = BlockedArray(G_init, repeat([L], R), repeat([L], R))

# G_init[Block(1, 3)] .= 0
# G_init[Block(3, 1)] .= 0

# G_init[Block(1, 4)] .= 0
# G_init[Block(4, 1)] .= 0

# G_init[Block(2, 3)] .= 0
# G_init[Block(3, 2)] .= 0

# G_init[Block(2, 4)] .= 0
# G_init[Block(4, 2)] .= 0

# G, Σ = SwapMatrix.schwinger_dyson(G_init, w, syk; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)

# blue = RGB(0,101/255,1)
# orange = RGB(1,154/255,0)
# grad = cgrad([blue, :gray95, orange], [0.0, 0.5, 1.0])
# p = heatmap(G, aspect_ratio = 1, clims=(-1, 1), yflip = true, color = grad, xaxis=false, yaxis=false, colorbar=false, grid=false, top_margin = -10Plots.cm, bottom_margin = -10Plots.cm, left_margin = -10Plots.cm, right_margin = -10Plots.cm, size=(512,512))

# # Permutation 3

# G_init = inv(SwapMatrix.differential(R * L; M = 1, periodic=false)./2) - I
# G_init = BlockedArray(G_init, repeat([L], R), repeat([L], R))

# G_init[Block(1, 2)] .= 0
# G_init[Block(2, 1)] .= 0

# G_init[Block(1, 3)] .= 0
# G_init[Block(3, 1)] .= 0

# G_init[Block(2, 4)] .= 0
# G_init[Block(4, 2)] .= 0

# G_init[Block(3, 4)] .= 0
# G_init[Block(4, 3)] .= 0

# G, Σ = SwapMatrix.schwinger_dyson(G_init, w, syk; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)

# blue = RGB(0,101/255,1)
# orange = RGB(1,154/255,0)
# grad = cgrad([blue, :gray95, orange], [0.0, 0.5, 1.0])
# p = heatmap(G, aspect_ratio = 1, clims=(-1, 1), yflip = true, color = grad, xaxis=false, yaxis=false, colorbar=false, grid=false, top_margin = -10Plots.cm, bottom_margin = -10Plots.cm, left_margin = -10Plots.cm, right_margin = -10Plots.cm, size=(512,512))
