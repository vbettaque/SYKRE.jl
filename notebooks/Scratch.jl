using CSV, DataFrames, Statistics, Plots, LinearAlgebra

include("../src/SYKRE.jl")
using .SYKRE
using .SYKRE.SYK
using .SYKRE.SYKMatrix
using .SYKRE.SREMatrix


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

# begin

# N = 1
# J = 1.
# q = 4
# M = 1
# L = 1000

# Ts = LinRange(0.5, 2, 20)

# βs = 1.0./Ts
# free_energies = zeros(length(βs))
# Σ_init = zeros(L, L)

# for i in reverse(eachindex(βs))
#     println(i)
#     β = βs[i]
#     syk = SYK.SYKData(N, J, q, M, β)
#     global free_energies[i], Σ_init = SYKMatrix.free_energy(L, syk; Σ_init=Σ_init)
# end

# p = plot(Ts, free_energies, label="J = 1, q = 4, L=1000")
# xaxis!("T")
# yaxis!("F(T)")

# plot!(Ts, -Ts * log(2) / 2 - 1/(4*q^2) * 1.0 ./ Ts, label="expectation")
# display(p)

# end

# begin

# N = 1
# J = 1.
# q = 4
# M = 1
# L = 2000

# Ts = LinRange(1, 10, 10)

# βs = 1.0./Ts
# free_energies = zeros(length(βs))
# Σ_init = zeros(L, L)

# for i in reverse(eachindex(βs))
#     println(i)
#     β = βs[i]
#     syk = SYK.SYKData(N, J, q, M, β)
#     global free_energies[i], Σ_init = SYKMatrix.free_energy(L, syk; Σ_init=Σ_init, max_iters=1000)
# end

# p = plot(Ts, free_energies, label="J = 1, q = 4, L=2000")
# xaxis!("T")
# yaxis!("F(T)")

# plot!(Ts, SYK.high_temp_free_energy.(βs, q, J), label="expectation")
# display(p)

# end

begin

N = 1
J = 1.
q = 4
M = 1
L = 500

Ts = LinRange(1, 10, 10)

βs = 1.0./Ts
renyi2s = zeros(length(βs))

Σ_β_init = zeros(L, L)
Σ_2β_init = zeros(4L, 4L)

for i in reverse(eachindex(βs))
    println(i)
    β = βs[i]
    syk = SYK.SYKData(N, J, q, M, β)
    global renyi2s[i], Σ_β_init, Σ_2β_init = SYKMatrix.renyi2(L, syk; Σ_β_init=Σ_β_init, Σ_2β_init=Σ_2β_init, max_iters=1000)
end

p = plot(Ts, renyi2s, label="J = 1, q = 4, L=1000")
xaxis!("T")
yaxis!("S_2")



end

D_minus = SREMatrix.differential(8; anti_periodic=true)
D_plus = SREMatrix.differential(8; anti_periodic=false)

G = inv(D_minus)
Σ = G.^3
prop_minus = D_minus - Σ
prop_plus = D_plus - Σ
inv(prop_plus)
det(prop_plus)
det(prop_minus)

inv(SREMatrix.differential(8; anti_periodic=true))
