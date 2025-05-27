using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier

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


N = 1
J = 1.
q = 4
M = 4
L= 2^13-1

β = 7.8

Δτ = β / L

syk_M = SYKData(N, J, q, M, β)
syk_2 = SYKData(N, J, q, 2, β)
syk_Z = SYKData(N, J, q, 1, β)

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

sre, Σ_M_real, Σ_2_real, Σ_Z_real = SREFourier.sre(L, syk_M; Σ_M_init=Σ_M_init, Σ_2_init=Σ_2_init, Σ_Z_init=Σ_Z_init,max_iters=100000)

Σ_M_freq = β * ifft(Σ_M_real, 1)
Σ_2_freq = β * ifft(Σ_2_real, 1)
Σ_Z_freq = β * ifft(Σ_Z_real, 1)

G_M_freq = SREFourier.G_SD_freq(Σ_M_freq, syk_M)
G_2_freq = SREFourier.G_SD_freq(Σ_2_freq, syk_2)
G_Z_freq = SYKFourier.G_SD_freq(Σ_Z_freq, syk_Z)

G_M_real = real(fft(G_M_freq, 1)) / β
G_2_real = real(fft(G_2_freq, 1)) / β
G_Z_real = real(fft(G_Z_freq, 1)) / β

τs = LinRange(0, β-Δτ, L)

p = plot(τs, G_2_real[:, 1, 1], label="11")
plot!(τs, G_2_real[:, 1, 2], label="12")
plot!(τs, G_2_real[:, 2, 1], label="21")
plot!(τs, G_2_real[:, 2, 2], label="22")

xlabel!("\\tau")
ylabel!("\$G_2\$")

display(p)

p = plot(τs, G_M_real[:, 1, 1], label="11")
plot!(τs, G_M_real[:, 2, 2], label="22")
plot!(τs, G_M_real[:, 3, 3], label="33")
plot!(τs, G_M_real[:, 4, 4], label="44")
plot!(τs, G_M_real[:, 1, 2], label="12")
plot!(τs, G_M_real[:, 2, 1], label="21")
plot!(τs, G_M_real[:, 1, 3], label="13")
plot!(τs, G_M_real[:, 3, 1], label="31")
plot!(τs, G_M_real[:, 1, 4], label="14")
plot!(τs, G_M_real[:, 4, 1], label="41")
plot!(τs, G_M_real[:, 2, 3], label="23")
plot!(τs, G_M_real[:, 3, 2], label="32")
plot!(τs, G_M_real[:, 3, 4], label="34")
plot!(τs, G_M_real[:, 4, 3], label="43")

xlabel!("\\tau")
ylabel!("\$G_4\$")

display(p)


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
