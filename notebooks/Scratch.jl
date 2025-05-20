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

# begin

# N = 1
# J = 1.
# q = 4
# M = 1
# L = 500

# Ts = LinRange(1, 10, 10)

# βs = 1.0./Ts
# renyi2s = zeros(length(βs))

# Σ_β_init = zeros(L, L)
# Σ_2β_init = zeros(4L, 4L)

# for i in reverse(eachindex(βs))
#     println(i)
#     β = βs[i]
#     syk = SYK.SYKData(N, J, q, M, β)
#     global renyi2s[i], Σ_β_init, Σ_2β_init = SYKMatrix.renyi2(L, syk; Σ_β_init=Σ_β_init, Σ_2β_init=Σ_2β_init, max_iters=1000)
# end

# p = plot(Ts, renyi2s, label="J = 1, q = 4, L=1000")
# xaxis!("T")
# yaxis!("S_2")



# end

# L = 1000
# L_rep = L÷2

# β = 1

# Δτ = β/L_rep

# syk = SYKData(1, 1, 4, 1, β)

# I_rep = [1 0; 0 1]

# D_minus = SREMatrix.differential(L_rep, anti_periodic=true)
# D_plus = SREMatrix.differential(L_rep; anti_periodic=false)

# D_minus_rep = kron(I_rep, D_minus)
# D_plus_rep = kron(I_rep, D_plus)

# purity, Σ = SYKMatrix.log_purity(L, syk)
# Σ

# prop_minus = D_minus_rep - Δτ^2 * Σ
# prop_plus = D_plus_rep - Δτ^2 * Σ

# pfaff_minus = sqrt(det(prop_minus))
# pfaff_plus = sqrt(det(prop_plus))

# p_plus = pfaff_plus / (pfaff_minus + pfaff_plus)

# G = p_plus * inv(prop_plus) + (1-p_plus) * inv(prop_minus)

# Σ_new = syk.J * G.^(syk.q-1)

# sqrt(det(D_minus_rep - Δτ^2 * Σ_new))
# sqrt(det(D_plus_rep - Δτ^2 * Σ_new))

# inv(SREMatrix.differential(8; anti_periodic=true))

# L = 2000
# β = 100
# syk2 = SYKData(1, 1, 4, 4, β)
# Σ_init = syk2.J * inv(SREMatrix.differential(L, anti_periodic=true))
# Σ, G,  = SREMatrix.schwinger_dyson(L, syk2, Σ_init=Σ_init)
# Σ

begin

N = 1
J = 1.
q = 4
M = 4
L = 2000

βs = LinRange(10, 20, 21)
sres = zeros(length(βs))

Σ_init = J * inv(SREMatrix.differential(L)).^(q-1)

Σ_M_init = Σ_init
Σ_2_init = Σ_init
Σ_Z_init = Σ_init

for i in eachindex(βs)
    println(i)
    β = βs[i]
    syk = SYK.SYKData(N, J, q, M, β)
    global sres[i], Σ_M_init, Σ_2_init, Σ_Z_init =
        SREMatrix.sre(L, syk; Σ_M_init=Σ_init, Σ_2_init=Σ_init, Σ_Z_init=Σ_init,max_iters=1000)
end

p = plot(βs, sres, label="J = 1, q = 4, L=4000")
xaxis!("β")
yaxis!("M")


end

# N = 1
# J = 1.
# q = 4
# M = 4
# L = 6000
# β = 100

# Σ_init = J * inv(SREMatrix.differential(L)).^(q-1)

# Σ_M_init = Σ_init
# Σ_2_init = Σ_init
# Σ_Z_init = Σ_init

# syk = SYK.SYKData(N, J, q, M, β)

# SREMatrix.sre(L, syk; Σ_M_init=Σ_M_init, Σ_2_init=Σ_2_init, Σ_Z_init=Σ_Z_init,max_iters=1000)

