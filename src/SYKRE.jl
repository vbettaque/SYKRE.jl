module SYKRE

using CSV, DataFrames, Statistics


include("MatrixSD.jl")
include("FourierSD.jl")


# function matrix_sres(L, βs, α, q, N, J; max_iters=10000)
#     steps = length(βs)
#     sres = zeros(steps)
#     # L, β, α. q, N, J
#     Threads.@threads for i=1:steps
#         println(i, " of ", steps)
#         sres[i] = MatrixSD.sre_saddlepoint(L, βs[i], α, q, N, J, max_iters=max_iters)
#         println("(T, SRE) = (", 1/βs[i], ", ", sres[i], ")")
#     end

#     return sres
# end


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

qs = [2, 4, 6, 8]
J = 1
α = 2
L = 500
T_start = 0.2
T_stop = 0.1
steps = 100
N = 1
max_iters = 10000

Ts = collect(LinRange(T_start, T_stop, steps))
βs = map(t -> 1/t, Ts)
Σ_init = SkewHermitian(zeros(L, L))

for i in eachindex(qs)
    println("q = ", qs[i])
    sres = MatrixSD.sre_saddlepoint(L, βs, α, qs[i], N, J, Σ_init=Σ_init, max_iters=max_iters)
    df = DataFrame(T = Ts, β = βs, sre = sres)
    file_name = string("./data/sykre_matrix_q", qs[i], ".csv")
    CSV.write(file_name, df)
end

using CSV, Plots, DataFrames
data_q2 = CSV.read("data/sykre_matrix_q2.csv", DataFrame)
data_q4 = CSV.read("data/sykre_matrix_q4.csv", DataFrame)
data_q6 = CSV.read("data/sykre_matrix_q6.csv", DataFrame)
plot(data_q2[!, 1], data_q2[!, 3], label="q=2")
plot!(data_q4[!, 1], data_q4[!, 3], label="q=4")
plot!(data_q6[!, 1], data_q6[!, 3], label="q=6")
xlabel!("T")
ylabel!("M2")

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
