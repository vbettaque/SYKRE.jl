using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier

function time_invariance(M, β)
    L, _ = size(M)
    Δτ = β / L
    τs = LinRange(0, 2β - Δτ, 2 * L)
    M_invariant = zeros(2 * L)
    M_invariant[1:L] = M[1:L, 1]
    M_invariant[L+2:2L] = reverse(M[1, 2:L])
    M_invariant[L+1] = (M_invariant[L] + M_invariant[L+2])/2
    return τs, M_invariant
end

function rand_time_invariant(L; periodic=false)
    A = zeros(L, L)
    rands = rand(Float64, L-1) / 4 .+ 0.25
    rands = (rands + reverse(rands)) / 2

    for i = 2:L
        for j = 1:(i-1)
            A[i, j] = rands[i - j]
        end
    end

    sign = periodic ? 1 : -1
    A += sign * A'
    A += I / 2
end

function rand_G_init(L, M)
    R = Matrix{Float64}(I, M, M)
    G_minus = rand_time_invariant(L; periodic = false)
    G_init = blockkron(R, G_minus)

    for _ = 2:M
        R = R[[M; 1:(M-1)], :]
        R[1, :] *= -1
        G_plus = rand_time_invariant(L; periodic = true)
        G_init += blockkron(R, G_plus)
    end

    return G_init
end

function generate_matrix_sre_data(L, M, q, βs)
    @assert iseven(M) && M > 0
    N = 1
    J = 1

    path = "data/sre_matrix/"
    filename = "sre" * string(M ÷ 2) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "β,sre,pf_minus_α,pf_plus_α,pf_minus_1,pf_plus_1\n")
    end

    G_M_0 = inv(SYKMatrix.differential(M * L)) - 0.5 * I
    G_M_0 = BlockedArray(G_M_0, repeat([L], M), repeat([L], M))
    G_M_init = G_M_0
    G_2_init = inv(SYKMatrix.differential(2 * L)) - 0.5 * I
    G_Z_init = inv(SYKMatrix.differential(L)) - 0.5 * I

    for i in eachindex(βs)
        β = βs[i]
        println(i, " out of ", length(βs), ": β = ", β)

        syk = SYKData(N, J, q, M, β)

        sre = 0
        pf_minus_M = 0
        pf_plus_M = 0
        pf_minus_2 = 0
        pf_plus_2 = 0

        t = 0.1

        sre, G_M, G_2, G_Z = SREMatrix.sre(G_M_init, G_2_init, G_Z_init, syk; params_M = (t, 2, 1000), params_2 = (t, 2, 1000), params_Z = (t, 2, 1000))
        Σ_M = SREMatrix.Σ_SD(G_M, syk)
        pf_minus_M, pf_plus_M = SREMatrix.pfaffians(Σ_M, syk)

        L_2, _ = size(G_2)
        syk2 = SYKData(syk.N, syk.J, syk.q, 2, syk.β)
        G_2_blocked = BlockedArray(G_2, [L_2 ÷ 2, L_2 ÷ 2], [L_2 ÷ 2, L_2 ÷ 2])
        Σ_2_blocked = SREMatrix.Σ_SD(G_2_blocked, syk2)
        pf_minus_2, pf_plus_2 = SREMatrix.pfaffians(Σ_2_blocked, syk2)

        # G_M_init = 0.5 * G_M + 0.5 * G_M_0
        # G_2_init = G_2
        # G_Z_init = G_Z

        df = DataFrame(β = β, sre = sre, pf_minus_α = pf_minus_M, pf_plus_α = pf_plus_M, pf_minus_1 = pf_minus_2, pf_plus_1 = pf_plus_2)
        CSV.write(file, df, append=true)
    end

end

# βs = LinRange(1, 30, 30)
βs = Float64.(5:20)
generate_matrix_sre_data(500, 4, 4, βs)


# data_2_1000 = CSV.File("data/sre_matrix/sre2_q2_L1000.csv"; select=["β", "sre"]) |> DataFrame
# data_4_1000 = CSV.File("data/sre_matrix/sre2_q4_L1000.csv"; select=["β", "sre"]) |> DataFrame
# data_6_1000 = CSV.File("data/sre_matrix/sre2_q6_L1000.csv"; select=["β", "sre"]) |> DataFrame

# data_4_2000 = CSV.File("data/sre_matrix/sre2_q4_L2000.csv"; select=["β", "sre"]) |> DataFrame
# data_6_2000 = CSV.File("data/sre_matrix/sre2_q6_L2000.csv"; select=["β", "sre"]) |> DataFrame

# p = plot(data_2_1000[:, 1], data_2_1000[:, 2], label="q=2, L = 1000")
# p = plot!(data_4_1000[:, 1], data_4_1000[:, 2], label="q=4, L = 1000")
# plot!(data_6_1000[:, 1], data_6_1000[:, 2], label="q=6, L = 1000")
# plot!(data_4_2000[:, 1], data_4_2000[:, 2], label="q=4, L = 2000")
# plot!(data_6_2000[:, 1], data_6_2000[:, 2], label="q=6, L = 2000")

# xlabel!("\$ β \$")
# ylabel!("\$M_2\$")
# ylims!(0, 1.5)
# xlims!(0, 30)

# display(p)

# N = 1
# J = 1.
# q = 6
# M = 4
# L = 1000
# β = 5

# syk = SYKData(N, J, q, M, β)

# R1 = Matrix{Float64}(I, M, M)
# R2 = R1[[M; 1:(M-1)], :]; R2[1, M] *= -1
# R3 = R2[[M; 1:(M-1)], :]; R3[1, M-1] *= -1
# R4 = R3[[M; 1:(M-1)], :]; R4[1, M-2] *= -1

# G1 = rand_time_invariant(L; periodic=false)
# G2 = rand_time_invariant(L; periodic=true)
# G3 = rand_time_invariant(L; periodic=true)
# G4 = rand_time_invariant(L; periodic=true)

# G_init = blockkron(R1, G1) + blockkron(R2, G2) + blockkron(R3, G3) + blockkron(R4, G4)

# G_init = inv(SREMatrix.differential(M * L; M=1, periodic=false))
# G_init = BlockedMatrix(G_init, repeat([L], M), repeat([L], M))

# plot(Gray.(G_init .- minimum(G_init)))

# G, Σ = SREMatrix.schwinger_dyson(G_init, syk; init_lerp = 0.5)

# plot(Gray.(G .- minimum(G)))


# τs = []
# G_blocks = blocks(G)
# G_functions = zeros(2 * L, M, M)


# p = plot()
# for i=1:M
#     for j=1:M
#         τs, G_functions[:, i, j] = time_invariance(G_blocks[i, j], β)
#         p = plot!(τs, G_functions[:, i, j], label="G_$(i)$(j)")
#     end
# end

# display(p)

#--------------------------------

# N = 1
# L = 1000
# J = 1
# q = 2
# M = 1

# βs = LinRange(0.1, 30, 31)
# Fs = zeros(length(βs))

# G_init = inv(SREMatrix.differential(L))
# for i = eachindex(βs)
#     syk = SYKData(N, J, q, M, βs[i])
#     G, Σ = SYKMatrix.schwinger_dyson(G_init, syk; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
#     Fs[i] = SYKMatrix.free_energy_saddle(G, Σ, syk)
#     G_init = G
# end

# p = plot(βs, Fs)
# plot!(βs, SYK.free_energy_q2.(βs, J; N=1))

# xaxis!("β")
# yaxis!("F")

# display(p)

#--------------------------------

# N = 1
# L = 1000
# J = 1
# q = 4
# M = 1

# βs = LinRange(1, 10, 10)
# purity_syk = zeros(length(βs))
# purity_sre = zeros(length(βs))

# G_init_syk = inv(SREMatrix.differential(L))
# G_init_sre = BlockedArray(G_init_syk, [L÷2, L÷2], [L÷2, L÷2])

# for i = eachindex(βs)
#     syk = SYKData(N, J, q, M, βs[i])
#     sre = SYKData(N, J, q, 2, βs[i])
#     purity_syk[i] = SYKMatrix.log2_purity_saddle(G_init_syk, syk; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
#     G_sre, Σ_sre = SREMatrix.schwinger_dyson(G_init_sre, sre; init_lerp = 0.1, lerp_divisor = 20, max_iters=1000)
#     purity_sre[i] = SREMatrix.log2_saddle(G_sre, Σ_sre, sre)
# end

# p = plot(βs, purity_syk)
# plot!(βs, purity_sre .- 0.5)

# xaxis!("β")
# yaxis!("log(purity)")

# display(p)
