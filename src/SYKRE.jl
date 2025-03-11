module SYKRE

using CSV, DataFrames, Statistics, SkewLinearAlgebra


include("MatrixSD.jl")


function matrix_sres(L, βs, α, q, N, J; max_iters=10000)
    steps = length(βs)
    sres = zeros(steps)
    # L, β, α. q, N, J
    Threads.@threads for i=1:steps
        println(i, " of ", steps)
        sres[i] = MatrixSD.sre_saddlepoint(L, βs[i], α, q, N, J, max_iters=max_iters)
        println("(T, SRE) = (", 1/βs[i], ", ", sres[i], ")")
    end

    return sres
end


function time_invariance(M, β)
    L, _ = size(M)
    Δτ = β / L
    Δs = collect(0:(L-1))
    M_avgs = zeros(L)
    M_stds = zeros(L)
    for k=1:L
        Δ = Δs[k]
        Ms = zeros(L)
        for i=1:L
            j = (i + Δ - 1) % L + 1
            Ms[i] = sign(j-i) * M[i, j]
        end
        M_avgs[k] = mean(Ms)
        M_stds[k] = std(Ms)
    end
    return Δτ * Δs, M_avgs, M_stds
end

q = 2
J = 1
α = 2
L = 400
T_min = 1
T_max = 2
steps = 100
N = 1
Σ_init = SkewHermitian(zeros(L, L))
max_iters = 10000

Ts = collect(LinRange(T_min, T_max, steps))
βs = map(t -> 1/t, Ts)
sres = MatrixSD.sre_saddlepoint(L, βs, α, q, N, J, Σ_init=Σ_init, max_iters=max_iters)

df = DataFrame(T = Ts, β = βs, sre = sres)
CSV.write("./data/test.csv", df)

using Plots
data = CSV.read("./data/test.csv", DataFrame)
plot(data[!, 1], data[!, 3])

# L = 100
# β = 1
# M = 2 * α
# Σ, G = MatrixSD.schwinger_dyson(L, β, M, q, J; sre=true, max_iters=10000)
# Δτs, G_means, G_stds = time_invariance(G, β)
# Δτs, Σ_means, Σ_stds = time_invariance(Σ, β)

# using Plots
# scatter(Δτs, G_means, yerr=G_stds, label="β = 1")
# xlabel!("k Δτ")
# ylabel!("G(τ, τ+kΔτ)/Δτ²")

# scatter(Δτs, Σ_means, yerr=Σ_stds, label="β = 1")
# xlabel!("k Δτ")
# ylabel!("Σ(τ, τ+kΔτ)/Δτ²")



end
