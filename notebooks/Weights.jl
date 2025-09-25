using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

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


function bisection(f, a, b; tol=0.0001, max_iter=100)
    @assert a < b
    fa = f(a)
    fb = f(b)
    @assert fa * fb < 0

    i = 1
    while i ≤ max_iter
        println("a = ", a, ", b = ", b)
        println("f(a) = ", fa, ", f(b) = ", fb)
        c = (a + b) / 2
        println("c = ", c)
        if (b - a) / 2 < tol
            return c, a, b
        end
        fc = f(c)
        println("f(c) = ", fc)
        if iszero(fc)
            return c, a, b
        end
        i += 1
        if sign(fc) == sign(fa)
            a = c
            fa = fc
        else
            b = c
            fb = fc
        end
    end
    println("Exceeded maximum number of steps!")
end

function weight_difference(w, L, syk::SYKData)
    G_init = Replicas.init(syk.M, L)

    t = min(inv(2 * syk.β * syk.J), 0.005)
    tol = max((syk.β * syk.J / L)^2 / 10, 1e-6)

    G, Σ = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = t, lerp_divisor = 2, tol=tol, max_iters=10000)

    pf_minus, pf_plus = WeightReplicas.pfaffians(Σ, syk)
    p_plus = pf_plus / (pf_plus + pf_minus)
    return p_plus - w
end


function critical_weight(L, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=10000)
    G_init = Replicas.init(syk.M, L)
    t = init_lerp
    w = 0

    G, Σ = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp / 10, lerp_divisor = lerp_divisor, tol=tol, max_iters=max_iters)
    pf_minus, pf_plus = WeightReplicas.pfaffians(Σ, syk)
    w_new = pf_plus / (pf_plus + pf_minus)

    for i = 1:max_iters
        w_lerp = t * w_new + (1 - t) * w

        G, Σ = WeightReplicas.schwinger_dyson(G_init, w_lerp, syk; init_lerp = init_lerp / 10, lerp_divisor = lerp_divisor, tol=tol, max_iters=max_iters)
        pf_minus, pf_plus = WeightReplicas.pfaffians(Σ, syk)
        w_iter = pf_plus / (pf_plus + pf_minus)

        err = (w_iter - w_lerp) / w_lerp

        if err < 0
            t /= lerp_divisor
            continue
        end

        if abs(err) < tol
            G, Σ = WeightReplicas.schwinger_dyson(G_init, w_iter, syk; init_lerp = init_lerp / 10, lerp_divisor = lerp_divisor, tol=tol, max_iters=max_iters)
            pf_minus, pf_plus = WeightReplicas.pfaffians(Σ, syk)
            w = pf_plus / (pf_plus + pf_minus)
            return w, G, Σ
        end

        w = w_iter
        G, Σ = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp / 10, lerp_divisor = lerp_divisor, tol=tol, max_iters=max_iters)
        pf_minus, pf_plus = WeightReplicas.pfaffians(Σ, syk)
        w_new = pf_plus / (pf_plus + pf_minus)
    end
end


###################################3

# N = 1
# J = 1
# q = 6
# M = 4
# L = 500

# βs = collect(5:5:400) / 10
# ws = zeros(length(βs))

# path = "data/sre_weights/"
# filename = "weights_M" * string(M) * "_q" * string(q) * "_L" * string(L) * ".csv"
# file = path * filename
# !ispath(path) && mkdir(path)
# if !isfile(file)
#     touch(file)
#     write(file, "β,weight,weight_lower,weight_upper\n")
# end

# w_min = 0.
# w_max = 0.15

# for i in eachindex(βs)
#     β = βs[i]
#     println(i, " out of ", length(ws), ": β = ", β)

#     syk = SYKData(N, J, q, M, β)
#     f(w) = weight_difference(w, L, syk)

#     w, w_lower, w_upper = bisection(f, w_min, w_max; tol=1e-5, max_iter=100)

#     df = DataFrame(β = β, weight = w, weight_lower = w_lower, weight_upper = w_upper)
#     CSV.write(file, df, append=true)

#     # global w_min = w_lower
# end

# weights_data = CSV.File("data/sre_weights/weights_M2_q4_L1000.csv") |> DataFrame

###########################################

# β = 4
# syk = SYKData(N, J, q, M, β)
# G_init = Replicas.init(M, L)
# PurityReplicas.schwinger_dyson(G_init, syk; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=10000)

# p = plot(weights_data[:,1], weights_data[:,2], label="q=4, M=2, L=1000")
# xlabel!("β")
# ylabel!("w_crit")
# display(p)

###########################################

# N = 1
# J = 1
# q = 4
# M = 4
# β = 5
# Ls = collect(1:50) .* 10
# weights = zeros(size(Ls))

# syk = SYKData(N, J, q, M, β)

# for i in eachindex(Ls)
#     L = Ls[i]
#     println(i, " out of ", length(Ls), ": L = ", L)
#     f(w) = weight_difference(w, L, syk)
#     w, w_lower, w_upper = bisection(f, 0, 0.5; tol=0.0001, max_iter=100)
#     weights[i] = w
# end

# p = plot(Ls, weights, label = "M = 4, q = 4, β = 5")
# xaxis!("L")
# yaxis!("weight")

# display(p)

###########################################

# N = 1
# J = 1
# q = 4
# M = 4
# β = 10
# L = 500

# ws = 0:100 / 100

# syk = SYKData(N, J, q, M, β)

# for i in eachindex(Ls)
#     L = Ls[i]
#     println(i, " out of ", length(Ls), ": L = ", L)
#     f(w) = weight_difference(w, L, syk)
#     w, w_lower, w_upper = bisection(f, 0, 0.5; tol=0.0001, max_iter=100)
#     weights[i] = w
# end

# p = plot(Ls, weights, label = "M = 4, q = 4, β = 5")
# xaxis!("L")
# yaxis!("weight")

# display(p)

###########################################

# N = 1
# J = 1
# q = 4
# M = 4
# β = 10
# L = 500

# syk = SYKData(N, J, q, M, β)

# critical_weight(L, syk; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=10000)

##############################################

# N = 1
# J = 1
# q = 4
# M = 4
# β = 10
# L = 500

# G_init_2 = Replicas.init(2, L)
# G_init_4 = Replicas.init(4, L)
# ws = LinRange(0, 0.05, 10)
# ps_2 = zeros(length(ws))
# ps_4 = zeros(length(ws))

# syk_2 = SYKData(N, J, q, 2, β)
# syk_4 = SYKData(N, J, q, 4, β)

# for i in eachindex(ws)
#     w = ws[i]
#     println(i, " out of ", length(ws), ": w = ", w)

#     G_2, Σ_2 = WeightReplicas.schwinger_dyson(G_init_2, w, syk_2; init_lerp = 0.01, lerp_divisor =2, tol=1e-5, max_iters=1000)
#     G_4, Σ_4 = WeightReplicas.schwinger_dyson(G_init_4, w, syk_4; init_lerp = 0.01, lerp_divisor =2, tol=1e-5, max_iters=1000)

#     pf_minus_2, pf_plus_2 = WeightReplicas.pfaffians(Σ_2, syk_2)
#     pf_minus_4, pf_plus_4 = WeightReplicas.pfaffians(Σ_4, syk_4)
#     ps_2[i] = pf_plus_2^2 / (pf_plus_2^2 + pf_minus_2^2)
#     ps_4[i] = pf_plus_4 / (pf_plus_4 + pf_minus_4)
# end

# p = plot(ws, ps_2, label="2b2")
# plot!(ws, ps_4, label = "1b4")
# plot!(ws, ws, label = "w")
# xaxis!("w")
# yaxis!("p")

# display(p)

#############################################################

function binary_entropy2(xs)
    entropy = zeros(length(xs))
    for i = eachindex(xs)
        @assert 0 ≤ xs[i] ≤ 1
        if !iszero(xs[i]) && !isone(xs[i])
            entropy[i] = - xs[i] * log2(xs[i]) - (1 - xs[i]) * log2(1 - xs[i])
        end
    end
    return entropy
end

N = 1
J = 1
q = 4
M = 4
β = 10
L = 500

G_init_1 = Replicas.init(1, 2*L)
G_init_2 = Replicas.init(2, L)
G_init_4 = Replicas.init(4, L)
# G_init_4 = blockkron([1 -0.5 0 0; -0.5 1 0 0; 0 0 1 -0.5; 0 0 -0.5 1], inv(SwapMatrix.differential(L)) - I)
# G_init_4 = BlockArray(G_init_4, [L, L, L , L], [L, L, L, L])
ws = LinRange(0, 1, 21)

entropies = binary_entropy2(ws)

saddles_1 = zeros(length(ws))
saddles_2 = zeros(length(ws))
saddles_4 = zeros(length(ws))

syk_1 = SYKData(N, J, q, 1, β)
syk_2 = SYKData(N, J, q, 2, β)
syk_4 = SYKData(N, J, q, 4, β)

G_init_1, _ = WeightReplicas.schwinger_dyson(G_init_1, 0, syk_1; init_lerp = 0.5, lerp_divisor = 2, tol=1e-10, max_iters=1000)

# @views G_init_1.blocks[:, :] = copy(G_init_seed.blocks[1:L, 1:L])

# @views G_init_2.blocks[:, :, 1] = copy(G_init_seed.blocks[1:L, 1:L])
# @views G_init_2.blocks[:, :, 2] = zeros(L, L) #copy(G_init_seed.blocks[L+1:2L, 1:L])

# @views G_init_4.blocks[:, :, 1] = copy(G_init_2.blocks[:, :, 1])
# @views G_init_4.blocks[:, :, 2] = copy(G_init_2.blocks[:, :, 2])
# @views G_init_4.blocks[:, :, 3] = zeros(L, L)
# @views G_init_4.blocks[:, :, 4] = copy(G_init_2.blocks[:, :, 2]')

for i in eachindex(ws)
    w = ws[i]
    println(i, " out of ", length(ws), ": w = ", w)

    G_1, Σ_1 = WeightReplicas.schwinger_dyson(G_init_1, w, syk_1; init_lerp = 0.5, lerp_divisor =2, tol=1e-10, max_iters=1000)

    @views G_init_2.blocks[:, :, 1] = copy(G_1.blocks[1:L, 1:L])
    @views G_init_2.blocks[:, :, 2] = copy(G_1.blocks[L+1:2L, 1:L])

    G_2, Σ_2 = WeightReplicas.schwinger_dyson(G_init_2, w, syk_2; init_lerp = 0.1, lerp_divisor =2, tol=1e-10, max_iters=1000)

    @views G_init_4.blocks[:, :, 1] = copy(G_2.blocks[:, :, 1])
    @views G_init_4.blocks[:, :, 2] = copy(G_2.blocks[:, :, 2])
    @views G_init_4.blocks[:, :, 3] = zeros(L, L)
    @views G_init_4.blocks[:, :, 4] = copy(G_2.blocks[:, :, 2]')

    G_4, Σ_4 = WeightReplicas.schwinger_dyson(G_init_4, w, syk_4; init_lerp = 0.1, lerp_divisor =2, tol=1e-10, max_iters=1000)
    # G_4, Σ_4 = SwapMatrix.schwinger_dyson(G_init_4, w, syk_4; init_lerp = 0.5, lerp_divisor =2, max_iters=1000)
    saddles_1[i] = 4 * WeightReplicas.log2_saddle(G_1, Σ_1, w, syk_1)
    saddles_2[i] = 2 * WeightReplicas.log2_saddle(G_2, Σ_2, w, syk_2)
    # saddles_4[i] = SwapMatrix.log2_saddle(G_4, Σ_4, w, syk_4)
    saddles_4[i] = WeightReplicas.log2_saddle(G_4, Σ_4, w, syk_4)
    # global G_init_2 = G_2
    # @views G_init_2.blocks[:, :, 1] -= Diagonal(G_init_2.blocks[:, :, 1])
    # global G_init_4 = G_4
    # @views G_init_4.blocks[:, :, 1] -= Diagonal(G_init_4.blocks[:, :, 1])
end

p = plot(ws, saddles_1 + entropies, label="4b1")
plot!(ws, saddles_2 + entropies, label = "2b2")
plot!(ws, saddles_4 + entropies, label = "1b4")
# plot!(ws, ws, label = "w")
xaxis!("w")
yaxis!("-I(w) + h(w)")

display(p)