using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier

# function time_invariance(M, β)
#     L, _ = size(M)
#     Δτ = β / L
#     Δs = collect(0:(L-1))
#     M_avgs = zeros(L)
#     M_stds = zeros(L)
#     for k=1:L
#         Δ = Δs[k]
#         Ms = zeros(L)
#         for i=1:L
#             j = (i + Δ - 1) % L + 1
#             Ms[i] = sign(j-i) * M[i, j]
#         end
#         M_avgs[k] = mean(Ms)
#         M_stds[k] = std(Ms)
#     end
#     display(M_avgs)
#     display(M_stds)
#     return Δτ * Δs, M_avgs, M_stds
# end

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

function generate_matrix_sre_data(L, α, q, βs)
    N = 1
    J = 1

    path = "data/sre_matrix/"
    filename = "sre" * string(α) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "β,sre,pf_minus_α,pf_plus_α,pf_minus_1,pf_plus_1\n")
    end

    Σ_init = J * inv(SREMatrix.differential(L)).^(q-1)

    for i in eachindex(βs)
        β = βs[i]
        println(i, ": β = ", β)
        syk = SYK.SYKData(N, J, q, 2*α, β)

        sre = 0
        pf_minus_α = 0
        pf_plus_α = 0
        pf_minus_1 = 0
        pf_plus_1 = 0

        try
            sre, Σ_M, Σ_2, Σ_Z =
                SREMatrix.sre(L, syk; Σ_M_init=Σ_init, Σ_2_init=Σ_init, Σ_Z_init=Σ_init,max_iters=500)

            pf_minus_α, pf_plus_α = SREMatrix.pfaffians(Σ_M, syk)
            pf_minus_1, pf_plus_1 = SREMatrix.pfaffians(Σ_2, SYK.SYKData(N, J, q, 2, β))

            display(Σ_2)

        catch e
            display(e)
            sre = NaN
            pf_minus_α = NaN
            pf_plus_α = NaN
            pf_minus_1 = NaN
            pf_plus_1 = NaN
        end

        df = DataFrame(β = β, sre = sre, pf_minus_α = pf_minus_α, pf_plus_α = pf_plus_α, pf_minus_1 = pf_minus_1, pf_plus_1 = pf_plus_1)
        CSV.write(file, df, append=true)
    end

end

function generate_fourier_sre_data(L, α, q, βs)
    N = 1
    J = 1
    M = 2 * α

    path = "data/sre_fourier/"
    filename = "sre" * string(α) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "β,sre,pf_minus_α,pf_plus_α,pf_minus_1,pf_plus_1\n")
    end

    G_diag_init = sign.(collect(1:L) .- (L-1)÷2)
    Σ_diag_init = J^2 * G_diag_init.^(q - 1)

    Σ_M_init = ones(L, M, M) / 32
    for i=1:M
        for j=1:i
            if i == j
                Σ_M_init[:, i, i] = copy(Σ_diag_init)
            else
                Σ_M_init[:, i, j] *= -1
            end
        end
    end

    Σ_2_init = ones(L, 2, 2) / 32
    Σ_2_init[:, 1, 1] = copy(Σ_diag_init)
    Σ_2_init[:, 2, 2] = copy(Σ_diag_init)
    Σ_2_init[:, 2, 1] *= -1

    Σ_Z_init = zeros(L, 1, 1)
    Σ_Z_init[:, 1, 1] = copy(Σ_diag_init)

    for i in eachindex(βs)
        β = βs[i]
        println(i, ": β = ", β)
        syk = SYK.SYKData(N, J, q, M, β)

        sre = 0
        pf_minus_α = 0
        pf_plus_α = 0
        pf_minus_1 = 0
        pf_plus_1 = 0

        sre, Σ_M, Σ_2, Σ_Z =
        SREFourier.sre(L, syk; Σ_M_init=Σ_M_init, Σ_2_init=Σ_2_init, Σ_Z_init=Σ_Z_init,max_iters=100000)

        pf_minus_α, pf_plus_α = SREFourier.pfaffians(Σ_M, syk)
        pf_minus_1, pf_plus_1 = SREFourier.pfaffians(Σ_2, SYK.SYKData(N, J, q, 2, β))

        Σ_Z_init = Σ_Z

        # try
        #     sre, Σ_M_init, Σ_2_init, Σ_Z_init =
        #         SREFourier.sre(L, syk; Σ_M_init=Σ_M_init, Σ_2_init=Σ_2_init, Σ_Z_init=Σ_Z_init,max_iters=1000)

        #     pf_minus_α, pf_plus_α = SREFourier.pfaffians(Σ_M_init, syk)
        #     pf_minus_1, pf_plus_1 = SREFourier.pfaffians(Σ_2_init, SYK.SYKData(N, J, q, 2, β))

        # catch e
        #     display(e)
        #     sre = NaN
        #     pf_minus_α = NaN
        #     pf_plus_α = NaN
        #     pf_minus_1 = NaN
        #     pf_plus_1 = NaN
        # end

        df = DataFrame(β = β, sre = sre, pf_minus_α = pf_minus_α, pf_plus_α = pf_plus_α, pf_minus_1 = pf_minus_1, pf_plus_1 = pf_plus_1)
        CSV.write(file, df, append=true)
    end

end

# generate_fourier_sre_data(2^13-1, 2, 4, LinRange(7.8, 10, 23))

# generate_sre_data(4000, 2, 2, LinRange(0.1, 10, 100))

# fftfreq(7, 7)

#--------------------------------------------

# N = 1
# J = 1.
# q = 4
# M = 4
# L= 2^13-1

# β = 8

# Δτ = β / L

# syk_M = SYKData(N, J, q, M, β)
# syk_2 = SYKData(N, J, q, 2, β)
# syk_Z = SYKData(N, J, q, 1, β)

# G_diag_init = -sign.(collect(1:L) .- (L-1)÷2) / 2
#     Σ_diag_init = J^2 * G_diag_init.^(q - 1)

#     Σ_M_init = ones(L, M, M) / 2
#     for i=1:M
#         for j=1:i
#             if i == j
#                 Σ_M_init[:, i, i] = copy(Σ_diag_init)
#             else
#                 Σ_M_init[:, i, j] *= -1
#             end
#         end
#     end

#     Σ_2_init = ones(L, 2, 2)
#     Σ_2_init[:, 1, 1] = copy(Σ_diag_init)
#     Σ_2_init[:, 2, 2] = copy(Σ_diag_init)
#     Σ_2_init[1:(L+1)÷2, 1, 2] = J^2 * LinRange(0.5, -0.5, (L+1)÷2).^(q-1)
#     Σ_2_init[(L+1)÷2:L, 1, 2] = J^2 * LinRange(-0.5, 0.5, (L+1)÷2).^(q-1)
#     Σ_2_init[:, 2, 1] = - Σ_2_init[:, 1, 2]

#     Σ_Z_init = zeros(L, 1, 1)
#     Σ_Z_init[:, 1, 1] = copy(Σ_diag_init)

# sre, Σ_M_real, Σ_2_real, Σ_Z_real = SREFourier.sre(L, syk_M; Σ_M_init=Σ_M_init, Σ_2_init=Σ_2_init, Σ_Z_init=Σ_Z_init,max_iters=100000)

# Σ_M_freq = β * ifft(Σ_M_real, 1)
# Σ_2_freq = β * ifft(Σ_2_real, 1)
# Σ_Z_freq = β * ifft(Σ_Z_real, 1)

# G_M_freq = SREFourier.G_SD_freq(Σ_M_freq, syk_M)
# G_2_freq = SREFourier.G_SD_freq(Σ_2_freq, syk_2)
# G_Z_freq = SYKFourier.G_SD_freq(Σ_Z_freq, syk_Z)

# G_M_real = real(fft(G_M_freq, 1)) / β
# G_2_real = real(fft(G_2_freq, 1)) / β
# G_Z_real = real(fft(G_Z_freq, 1)) / β

# τs = LinRange(0, 2*β-Δτ, L)

# p = plot(τs, G_2_real[:, 1, 1], label="11")
# plot!(τs, G_2_real[:, 1, 2], label="12")
# plot!(τs, G_2_real[:, 2, 1], label="21")
# plot!(τs, G_2_real[:, 2, 2], label="22")

# xlabel!("\\tau")
# ylabel!("\$G_2\$")

# display(p)

# p = plot(τs, G_M_real[:, 1, 1], label="11 = $(round(mean( G_M_real[:, 1, 1]), digits=3))")
# plot!(τs, G_M_real[:, 2, 2], label="22 = $(round(mean( G_M_real[:, 2, 2]), digits=3))")
# plot!(τs, G_M_real[:, 3, 3], label="33 = $(round(mean( G_M_real[:, 3, 3]), digits=3))")
# plot!(τs, G_M_real[:, 4, 4], label="44 = $(round(mean( G_M_real[:, 4, 4]), digits=3))")
# plot!(τs, G_M_real[:, 1, 2], label="12 = $(round(mean( G_M_real[:, 1, 2]), digits=3))")
# plot!(τs, G_M_real[:, 2, 1], label="21 = $(round(mean( G_M_real[:, 2, 1]), digits=3))")
# plot!(τs, G_M_real[:, 1, 3], label="13 = $(round(mean( G_M_real[:, 1, 3]), digits=3))")
# plot!(τs, G_M_real[:, 3, 1], label="31 = $(round(mean( G_M_real[:, 3, 1]), digits=3))")
# plot!(τs, G_M_real[:, 1, 4], label="14 = $(round(mean( G_M_real[:, 1, 4]), digits=3))")
# plot!(τs, G_M_real[:, 4, 1], label="41 = $(round(mean( G_M_real[:, 4, 1]), digits=3))")
# plot!(τs, G_M_real[:, 2, 3], label="23 = $(round(mean( G_M_real[:, 2, 3]), digits=3))")
# plot!(τs, G_M_real[:, 3, 2], label="32 = $(round(mean( G_M_real[:, 3, 2]), digits=3))")
# plot!(τs, G_M_real[:, 2, 4], label="24 = $(round(mean( G_M_real[:, 2, 4]), digits=3))")
# plot!(τs, G_M_real[:, 4, 2], label="42 = $(round(mean( G_M_real[:, 4, 2]), digits=3))")
# plot!(τs, G_M_real[:, 3, 4], label="34 = $(round(mean( G_M_real[:, 3, 4]), digits=3))")
# plot!(τs, G_M_real[:, 4, 3], label="43 = $(round(mean( G_M_real[:, 4, 3]), digits=3))")

# xlabel!("\\tau")
# ylabel!("\$G_4\$")

# display(p)

#--------------------------------------------
#
# N = 1
# J = 1.
# q = 4
# M = 4
# L= 4000

# β = 50

# syk_M = SYKData(N, J, q, M, β)
# syk_2 = SYKData(N, J, q, 2, β)
# syk_Z = SYKData(N, J, q, 1, β)

# L_rep_M = L ÷ M
# L_rep_2 = L ÷ 2

# Σ_init = J * inv(SREMatrix.differential(L)).^(q-1)

# sre, Σ_M, Σ_2, Σ_Z = SREMatrix.sre(L, syk_M; Σ_M_init=Σ_init, Σ_2_init=Σ_init, Σ_Z_init=Σ_init,max_iters=1000)

# G_M = SREMatrix.G_SD(Σ_M, syk_M)
# G_2 = SREMatrix.G_SD(Σ_2, syk_2)
# G_Z = SYKMatrix.G_SD(Σ_Z, syk_Z)

# Δτ_M = β / L_rep_M
# τs_M = LinRange(0, 2*β - Δτ_M, 2 * L_rep_M)
# G_M_invariant = zeros(2 * L_rep_M, M, M)

# Δτ_2 = β / L_rep_2
# τs_2 = LinRange(0, 2*β - Δτ_2, 2 * L_rep_2)
# G_2_invariant = zeros(2 * L_rep_2, 2, 2)

# for i=1:M
#     for j=1:M
#         i_range = ((i-1) * L_rep_M + 1):(i * L_rep_M)
#         j_range = ((j-1) * L_rep_M + 1):(j * L_rep_M)
#         _, G_M_invariant[(L_rep_M+1):(2*L_rep_M), i, j], _ = time_invariance(G_M[i_range, j_range], β)
#         G_M_invariant[1:L_rep_M, i, j] = -G_M_invariant[(L_rep_M+1):(2*L_rep_M), i, j]
#     end
# end

# for i=1:2
#     for j=1:2
#         i_range = ((i-1) * L_rep_2 + 1):(i * L_rep_2)
#         j_range = ((j-1) * L_rep_2 + 1):(j * L_rep_2)
#          _, G_2_invariant[(L_rep_2+1):(2*L_rep_2), i, j], _ = time_invariance(G_2[i_range, j_range], β)
#         G_2_invariant[1:L_rep_2, i, j] = -G_2_invariant[(L_rep_2+1):(2*L_rep_2), i, j]
#     end
# end

# p = plot(τs_2, G_2_invariant[:, 1, 1], label="11")
# plot!(τs_2, G_2_invariant[:, 1, 2], label="12")
# plot!(τs_2, G_2_invariant[:, 2, 1], label="21")
# plot!(τs_2, G_2_invariant[:, 2, 2], label="22")

# xlabel!("\\tau")
# ylabel!("\$G_2\$")

# display(p)

# p = plot(τs_M, G_M_invariant[:, 1, 1], label="11")
# plot!(τs_M, G_M_invariant[:, 2, 2], label="22")
# plot!(τs_M, G_M_invariant[:, 3, 3], label="33")
# plot!(τs_M, G_M_invariant[:, 4, 4], label="44")
# plot!(τs_M, G_M_invariant[:, 1, 2], label="12")
# plot!(τs_M, G_M_invariant[:, 2, 1], label="21")
# plot!(τs_M, G_M_invariant[:, 1, 3], label="13")
# plot!(τs_M, G_M_invariant[:, 3, 1], label="31")
# plot!(τs_M, G_M_invariant[:, 1, 4], label="14")
# plot!(τs_M, G_M_invariant[:, 4, 1], label="41")
# plot!(τs_M, G_M_invariant[:, 2, 3], label="23")
# plot!(τs_M, G_M_invariant[:, 3, 2], label="32")
# plot!(τs_M, G_M_invariant[:, 3, 4], label="34")
# plot!(τs_M, G_M_invariant[:, 4, 3], label="43")

# xlabel!("\\tau")
# ylabel!("\$G_4\$")

# display(p)

#--------------------------------------------

# N = 1
# J = 1.
# q = 4
# M = 4

# L_freq = 2^12-1
# L_matrix = 2000

# βs = LinRange(0.1, 5, 10)
# purities_syk = zeros(length(βs))
# purities_sre = zeros(length(βs))
# purities_matrix = zeros(length(βs))

# G_diag_init = -sign.(collect(1:L_freq) .- (L_freq-1)÷2) / 2
# Σ_diag_init = J^2 * G_diag_init.^(q - 1)

# Σ_init_syk = zeros(L_freq, 1, 1)
# Σ_init_syk[:, 1, 1] = copy(Σ_diag_init)

# Σ_init_sre = zeros(L_freq, 2, 2)
# Σ_init_sre[:, 1, 1] = copy(Σ_diag_init)
# Σ_init_sre[:, 2, 2] = copy(Σ_diag_init)
# # Σ_init_sre[1:(L+1)÷2, 1, 2] = J^2 * LinRange(0.5, -0.5, (L+1)÷2).^(q-1)
# # Σ_init_sre[(L+1)÷2:L, 1, 2] = J^2 * LinRange(-0.5, 0.5, (L+1)÷2).^(q-1)
# Σ_init_sre[:, 1, 2] = J^2 * (ones(L_freq) / 2).^(q-1)
# Σ_init_sre[:, 2, 1] = - Σ_init_sre[:, 1, 2]

# Σ_init_matrix = J^2 * inv(SREMatrix.differential(L_matrix)).^(q-1)

# for i in eachindex(βs)
#     syk_β = SYKData(N, J, q, 2, βs[i])
#     syk_2β = SYKData(N, J, q, 1, 2 * βs[i])
#     purities_syk[i], _ =  SYKFourier.logZ(L_freq, syk_2β; Σ_init=Σ_init_syk)
#     purities_sre[i], _ =  SREFourier.logZ(L_freq, syk_β; Σ_init=Σ_init_sre)
#     purities_sre[i] -= log(2)/2
#     purities_matrix[i], _ = SREMatrix.logZ(L_matrix, syk_β; Σ_init=Σ_init_matrix)
#     purities_matrix[i] -= log(2)/2
# end

# p = plot(βs, purities_syk, label="syk")
# plot!(βs, purities_sre, label="sre_freq")
# plot!(βs, purities_matrix, label="sre_matrix")
# xlabel!("β")
# ylabel!("log(purity)")

# display(p)


# βs = LinRange(0.1, 2.5, 25)
# free_energies_freq = zeros(length(βs))
# free_energies_matrix = zeros(length(βs))

# G_diag_init = sign.(collect(1:L_freq) .- (L_freq-1)÷2)
# Σ_diag_init = J^2 * G_diag_init.^(q - 1)
# Σ_init_freq = zeros(L_freq, 1, 1)
# Σ_init_freq[:, 1, 1] = copy(Σ_diag_init)

# # Σ_init_freq = zeros(L_freq, M, M)
# Σ_init_matrix = zeros(L_matrix, L_matrix)

# for i in reverse(eachindex(βs))
#     syk = SYKData(N, J, q, M, βs[i])
#     global free_energies_freq[i], _ =  SYKFourier.free_energy(L_freq, syk; Σ_init=Σ_init_freq)
#     global free_energies_matrix[i], Σ_init_matrix =  SYKMatrix.free_energy(L_matrix, syk; Σ_init=Σ_init_matrix)
# end

# β, Σ_inv = time_invariance(Σ_init_matrix, 10)

# plot(β, -Σ_inv)

# p = plot(1:L_freq÷2, [sum(Σ_init_freq[i, :, :]) for i=1:L_freq÷2])

# display(p)

# Ts = 1.0 ./ βs

# p = plot(Ts, free_energies_freq, label="Fourier")
# plot!(Ts, free_energies_matrix, label="Matrix")
# plot!(Ts, SYK.free_energy_q2.(1.0 ./ Ts, 1), label="exact")
# xlabel!("T")
# ylabel!("F")

#--------------------------------------------

# N = 1
# J = 1.
# q = 4
# M = 2
# L_freq = L_freq = 2^13-1
# L_matrix = 2000
# β = 4

# L_rep_matrix = L_matrix ÷ 2

# G_init_matrix = (inv(SREMatrix.differential(L_matrix)))


# Σ_init_matrix = J^2 * G_init_matrix.^(q-1)

# Σ_init_matrix_inv = zeros(L_matrix, 2, 2)

# for i=1:2
#     for j=1:2
#         i_range = ((i-1) * L_rep_matrix + 1):(i * L_rep_matrix)
#         j_range = ((j-1) * L_rep_matrix + 1):(j * L_rep_matrix)
#         _, Σ_init_matrix_inv[:, i, j] = time_invariance(Σ_init_matrix[i_range, j_range], β)
#     end
# end

# freqs = fftfreq(L_freq, π * L_freq)
# odd = isodd.(fftfreq(L_freq, L_freq))
# G_init_diag_freq = zeros(ComplexF64, L_freq)
# G_init_diag_freq[2:end] = 0.25 * im ./ freqs[2:end]
# G_init_diag = real(fft(G_init_diag_freq))
# # plot(1:L_freq, real(G_init_diag))
# Σ_init_diag = J^2 * G_init_diag

# # plot(1:L_freq, Σ_init_diag[1:end])

# Σ_init_freq = zeros(L_freq, 2, 2)
# Σ_init_freq[:, 1, 1] = copy(Σ_init_diag)
# Σ_init_freq[:, 2, 2] = copy(Σ_init_diag)
# Σ_init_freq[:, 1, 2] = -J^2 * (ones(L_freq) / 2).^(q-1)
# Σ_init_freq[:, 2, 1] = -copy(Σ_init_freq[:, 1, 2])

# βs = LinRange(0.1, 10, 300)
# pf_plus_freq = zeros(size(βs))
# pf_minus_freq = zeros(size(βs))
# pf_plus_matrix = zeros(size(βs))
# pf_minus_matrix = zeros(size(βs))
# pf_plus_matrix_inv = zeros(size(βs))
# pf_minus_matrix_inv = zeros(size(βs))

# for i = eachindex(βs)
#     syk_2 = SYKData(N, J, q, 2, βs[i])
#     I_rep = Matrix{Float64}(I, 2, 2)
#     # println("det(Δ_- - Σ_matrix) = ", det(kron(I_rep, SREMatrix.differential(L_matrix÷2)) - (2*βs[i]/L_matrix)^2 * Σ_init_matrix))
#     # println("det(Δ_+ - Σ_matrix) = ", det(kron(I_rep, SREMatrix.differential(L_matrix÷2, anti_periodic=false)) - (2*βs[i]/L_matrix)^2 * Σ_init_matrix))
#     ωs = fftfreq(L_freq, π * L_freq / βs[i])
#     Σ_freq = βs[i] * ifft(Σ_init_freq, 1)
#     dets = [det(Σ_freq[i, :, :]) for i=1:L_freq]
#     # println("log det = ", reduce(+, log.(dets)))
#     det_freq = reduce(*, [det(Σ_freq[i, :, :]) for i=1:L_freq])
#     # println("det(Σ_freq) = ", det_freq)

#     pf_minus_matrix[i], pf_plus_matrix[i] = SREMatrix.pfaffians(Σ_init_matrix, syk_2)
#     pf_minus_freq[i], pf_plus_freq[i] = SREFourier.pfaffians(Σ_init_freq, syk_2)
#     pf_minus_matrix_inv[i], pf_plus_matrix_inv[i] = SREFourier.pfaffians(Σ_init_matrix_inv[2:end, :, :], syk_2)
# end


# p = plot(βs, pf_minus_matrix, label="matrix")
# plot!(βs, pf_minus_freq, label="freq")
# plot!(βs, pf_minus_matrix_inv, label="matrix as freq")
# display(p)

# p = plot(βs, pf_plus_matrix, label="matrix")
# plot!(βs, pf_plus_freq, label="freq")
# plot!(βs, pf_plus_matrix_inv, label="matrix as freq")
# plot!(βs, sqrt.(4 * sinh.(βs / 4).^4), label="right analytic")
# plot!(βs, sqrt.((βs).^4 / 64), label="wrong analytic")
# ylims!(0, 100)
# display(p)

# p = plot(βs, pf_plus_matrix ./ pf_plus_freq, label="ratio")
# plot!(βs, cosh.(βs/9.9).^4, label="\$cosh(\\beta/8.9)^4\$")

# display(p)

# p = plot(βs, pf_plus_matrix - pf_plus_freq, label="difference")
# plot!(βs, sinh.(βs/5.2).^4, label="\$cosh(\\beta/8.9)^4\$")
# display(p)

# p = plot(βs, pf_plus_matrix ./ pf_minus_matrix, label="matrix ratio")
# plot!(βs, 1.0 ./ (1 .+ (2^(q/2) ./ (βs * J)).^2), label="eom")

# p = plot(βs, pf_plus_freq ./ pf_minus_freq, label="freq ratio")
# plot!(βs, 1.0 ./ (1 .+ (2^(q/2) ./ (βs * J)).^2), label="eom")


# display(p)

#--------------------------------------------

N = 1
J = 1.
q = 4
M = 4
L_freq = L_freq = 2^11-1
L_matrix = 2000
β = 4

ωs = fftfreq(L_freq, π * L_freq / β)
odd = isodd.(fftfreq(L_freq, L_freq))

syk_2 = SYKData(N, J, q, 2, β)

G_init_matrix = inv(SREMatrix.differential(L_matrix))
Σ_init_matrix = J^2 * G_init_matrix.^(q-1)

G_init_diag = zeros(ComplexF64, L_freq)
G_init_diag[2:end] = 1.0 ./(-im * ωs[2:end])
G_init_diag = real(fft(G_init_diag, 1)) / β
Σ_init_diag = J^2 * G_init_diag.^(q-1)

Σ_init_off = zeros(ComplexF64, L_freq)
Σ_init_off = 2 * J^2 ./ sqrt.(4 * J^2 .+ ωs.^2)
Σ_init_off = real(fft(Σ_init_off, 1)) / β


plot(1:L_freq, Σ_init_off)

Σ_init_freq = zeros(L_freq, 2, 2)
Σ_init_freq[:, 1, 1] = copy(Σ_init_diag)
Σ_init_freq[:, 2, 2] = copy(Σ_init_diag)
Σ_init_freq[:, 1, 2] = -copy(Σ_init_off)
Σ_init_freq[:, 2, 1] = copy(Σ_init_off)

sre_matrix, Σ_2_matrix, G_2_matrix = SREMatrix.logZ(L_matrix, syk_2; Σ_init=Σ_init_matrix, max_iters=1000)
sre_freq, Σ_2_freq, G_2_freq = SREFourier.logZ(L_freq, syk_2; Σ_init=Σ_init_freq, max_iters=100000)

Δτ_matrix = β / L_matrix
L_rep_matrix = L_matrix ÷ 2
τs_matrix = LinRange(0, 2*β - Δτ_matrix, 2 * L_rep_matrix)
Σ_2_matrix_inv = zeros(2 * L_rep_matrix, 2, 2)
G_2_matrix_inv = zeros(2 * L_rep_matrix, 2, 2)

for i=1:2
    for j=1:2
        i_range = ((i-1) * L_rep_matrix + 1):(i * L_rep_matrix)
        j_range = ((j-1) * L_rep_matrix + 1):(j * L_rep_matrix)
        _, Σ_2_matrix_inv[:, i, j] = time_invariance(Σ_2_matrix[i_range, j_range], β)
        _, G_2_matrix_inv[:, i, j] = time_invariance(G_2_matrix[i_range, j_range], β)
    end
end

Δτ_freq = β / L_freq
τs_freq = LinRange(0, 2*β - Δτ_freq, L_freq)

p = plot(τs_matrix, G_2_matrix_inv[:, 1, 1], label="matrix_11")
plot!(τs_matrix, G_2_matrix_inv[:, 1, 2], label="matrix_12")
plot!(τs_matrix, G_2_matrix_inv[:, 2, 1], label="matrix_21")
plot!(τs_matrix, G_2_matrix_inv[:, 2, 2], label="matrix_22")

xlabel!("\\tau")
ylabel!("\$G_2\$")
ylims!(-0.6, 0.6)

display(p)

p = plot(τs_freq, G_2_freq[:, 1, 1], label="freq_11")
plot!(τs_freq, G_2_freq[:, 1, 2], label="freq_12")
plot!(τs_freq, G_2_freq[:, 2, 1], label="freq_21")
plot!(τs_freq, G_2_freq[:, 2, 2], label="freq_22")

xlabel!("\\tau")
ylabel!("\$G_2\$")
ylims!(-0.6, 0.6)

pf_minus_matrix, pf_plus_matrix = SREMatrix.pfaffians(Σ_2_matrix, syk_2)
pf_plus_matrix / (pf_minus_matrix + pf_plus_matrix)
pf_minus_inv, pf_plus_inv = SREFourier.pfaffians(Σ_2_matrix_inv[2:end, :, :], syk_2)
pf_plus_inv / (pf_plus_inv + pf_minus_inv)

display(p)

# ------------------------------------

# N = 1
# J = 1.
# q = 4
# M = 2
# L = 2000
# L_rep = L ÷ 2
# βs = LinRange(0.1, 5, 10)

# G_init = inv(SREMatrix.differential(L))
# Σ_init = J^2 * G_init.^(q-1)

# pf_minus_matrix = zeros(length(βs))
# pf_plus_matrix = zeros(length(βs))
# pf_minus_freq = zeros(length(βs))
# pf_plus_freq = zeros(length(βs))

# for i = eachindex(βs)
#     syk_2 = SYKData(N, J, q, 2, βs[i])
#     sre, Σ_2, G_2 = SREMatrix.logZ(L, syk_2; Σ_init=Σ_init, max_iters=1000)
#     pf_minus_matrix[i], pf_plus_matrix[i] = SREMatrix.pfaffians(Σ_2, syk_2)
#     Σ_2_inv = zeros(L, 2, 2)
#     for i=1:2
#         for j=1:2
#             i_range = ((i-1) * L_rep + 1):(i * L_rep)
#             j_range = ((j-1) * L_rep + 1):(j * L_rep)
#             _, Σ_2_inv[:, i, j] = time_invariance(Σ_2[i_range, j_range], βs[i])
#         end
#     end
#     pf_minus_freq[i], pf_plus_freq[i] = SREFourier.pfaffians(Σ_2_inv[2:end, :, :], syk_2)
# end

# p = plot(βs, pf_minus_matrix, label="matrix")
# plot!(βs, pf_minus_freq, label="invariant")

# xlabel!("\\beta")
# ylabel!("\$pf_-\$")

# display(p)

# p = plot(βs, pf_plus_matrix, label="matrix")
# plot!(βs, pf_plus_freq, label="invariant")

# xlabel!("\\beta")
# ylabel!("\$pf_+\$")

# display(p)

# p_matrix = pf_minus_matrix ./  (pf_minus_matrix + pf_plus_matrix)
# pf_freq = pf_minus_freq ./  (pf_minus_freq + pf_plus_freq)

# p = plot(βs, pf_minus_matrix ./  (pf_minus_matrix + pf_plus_matrix), label="matrix")
# plot!(βs, pf_minus_freq ./  (pf_minus_freq + pf_plus_freq), label="freq")
# # plot!(βs, cosh.(βs/9).^2, label="\$cosh(\\beta/8.9)^4\$")

# xlabel!("\\beta")
# ylabel!("\$pf_-\$")

# display(p)
