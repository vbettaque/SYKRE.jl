using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier

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

function generate_sre_data(L, α, q, βs)
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

# generate_sre_data(4000, 2, 6, LinRange(0.1, 10, 100))

# generate_sre_data(8, 2, 4, [0.01])


N = 1
J = 1.
q = 2
M = 1
L_freq = 6000
L_matrix = 2000

Ts = LinRange(0.1, 1, 10)
βs = 1.0 ./ Ts
free_energies_freq = zeros(length(βs))
free_energies_matrix = zeros(length(βs))

Σ_init_freq = zeros(L_freq, M, M)
Σ_init_matrix = zeros(L_matrix, L_matrix)

for i in reverse(eachindex(βs))
    syk = SYKData(N, J, q, M, βs[i])
    global free_energies_freq[i], Σ_init_freq =  SYKFourier.free_energy(L_freq, syk; Σ_init=Σ_init_freq)
    global free_energies_matrix[i], Σ_init_matrix =  SYKMatrix.free_energy(L_matrix, syk; Σ_init=Σ_init_matrix)
end

β, Σ_inv = time_invariance(Σ_init_matrix, 10)

# plot(β, -Σ_inv)

# p = plot(1:L_freq÷2, [sum(Σ_init_freq[i, :, :]) for i=1:L_freq÷2])

# display(p)


p = plot(Ts, free_energies_freq, label="Fourier")
plot!(Ts, free_energies_matrix, label="Matrix")
plot!(Ts, SYK.free_energy_q2.(1.0 ./ Ts, 1), label="exact")
xlabel!("T")
ylabel!("F")


free_energy_q2(β, J; N=1)
