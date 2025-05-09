module SYKRE

using CSV, DataFrames, Statistics

include("SYK.jl")
include("SYKMatrix.jl")
include("SREMatrix.jl")
include("SYKFourier.jl")
include("SREFourier.jl")


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

# qs = [2, 4, 6, 8]
# J = 1
# α = 2
# L = 500
# T_start = 0.2
# T_stop = 0.1
# steps = 100
# N = 1
# max_iters = 10000

# Ts = collect(LinRange(T_start, T_stop, steps))
# βs = map(t -> 1/t, Ts)
# Σ_init = SkewHermitian(zeros(L, L))

# for i in eachindex(qs)
#     println("q = ", qs[i])
#     sres = MatrixSD.sre_saddlepoint(L, βs, α, qs[i], N, J, Σ_init=Σ_init, max_iters=max_iters)
#     df = DataFrame(T = Ts, β = βs, sre = sres)
#     file_name = string("./data/sykre_matrix_q", qs[i], ".csv")
#     CSV.write(file_name, df)
# end

# using CSV, Plots, DataFrames
# data_q2 = CSV.read("data/sykre_matrix_q2.csv", DataFrame)
# data_q4 = CSV.read("data/sykre_matrix_q4.csv", DataFrame)
# data_q6 = CSV.read("data/sykre_matrix_q6.csv", DataFrame)
# plot(data_q2[!, 2], data_q2[!, 3], label="q=2")
# plot!(data_q4[!, 2], data_q4[!, 3], label="q=4")
# plot!(data_q6[!, 2], data_q6[!, 3], label="q=6")
# xlabel!("T")
# ylabel!("M2")

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

# using FFTW
# using Plots

# N = 1
# β = 1
# J = 1.
# q = 4
# M = 4
# L = 1000

# syk = SYK.SYKData(N, J, q, M, β)

# Σ, G = SYKMatrix.schwinger_dyson(L, syk)
# Σ_, G_ = SYKFourier.schwinger_dyson(2 * L, syk)

# τs, G_avg, _ = time_invariance(G, β)

# p = plot(τs[2:2:L], -G_avg[2:2:L])
# plot!(τs, G_[1:L])
# ylims!((-1, 1))
# display(p)


# L = 8
# using LinearAlgebra
# using SkewLinearAlgebra


# D_minus = SREMatrix.differential(L; anti_periodic=true)
# D_plus = SREMatrix.differential(L; anti_periodic=false)

# eta = D_minus - D_plus

# Σ = skewhermitian(rand(Float64, L, L))
# pfaffian(Σ)
# Σ
# replica_I = Matrix{Float64}(I, 2, 2)

# replica_corrs = [1 0.5; 0.5 1]

# D_plus_replica = kron(replica_I, D_plus)
# Σ_replica = kron(replica_corrs, Σ)

# pfaffian(D_plus_replica - Σ_replica)
# pfaffian(Σ_replica)

# pfaffian(D_plus - Σ)

# display(D)

# Δτ = 2β / L
# τs = collect((0:L-1) * Δτ)


# # Σ_init = (sign.(τs) + sign.(τs .- β) - 2 * τs / β .+ 1).^(q-1)
# Σ_fft = zeros(ComplexF64, L)
# Σ_fft[1] = -1
# Σ_init = real(fft(Σ_fft)) / (2 * syk.β)

using LinearAlgebra
using SkewLinearAlgebra

N = 1
β = 2
J = 1.
q = 4
M = 4
L = 6

D_minus = SREMatrix.differential(L; anti_periodic=true)
D_plus = SREMatrix.differential(L; anti_periodic=false)

replica_I = Matrix{Float64}(I, 2, 2)

replica_corrs = [1 0.5; 0.5 1]

syk = SYK.SYKData(N, J, q, M, β)

Δτ = syk.β/L
Σ_init = (Δτ)^2 * SkewHermitian(syk.J^2 * map(g -> g^(syk.q-1), inv(D_minus)))

Σ, G = SREMatrix.schwinger_dyson(L, syk; Σ_init = Σ_init)

Σ
pfaffian(Σ)
pfaffian(D_plus - Σ_init)
pfaffian(D_minus - Σ_init)
pfaffian(kron(replica_I, D_plus) - Δτ^2 * kron(replica_corrs, Σ))

using Plots
τs, G_avg, _ = time_invariance(G, syk.β)

p = plot(τs[2:2:L], -G_avg[2:2:L])
ylims!(0, 0.5)
display(p)

D_rep = kron(replica_I, D_minus)

inv(D_rep)

(D_minus - D_plus)/2

D_minus
D_plus



L = 200
D_minus = SREMatrix.differential(L; anti_periodic=true)
D_plus = SREMatrix.differential(L; anti_periodic=false)
eta = D_plus - D_minus
pfaffian(D_plus - eta)
pfaffian(D_minus - eta)
end
