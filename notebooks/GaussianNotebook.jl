using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, ProgressMeter

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.Gaussian


# N = 100
# J = 1
# samples = 1000

# βs = collect(0:1:20)
# ws = zeros(length(βs))

# H = Gaussian.rand_hamiltonian(N, J)

# for i = eachindex(βs)
#     β = βs[i]
#     println("β = ", β)
#     Γ = 2 * tanh(β * H / 2)
#     w_means = zeros(samples)
#     ps = zeros(samples)
#     Threads.@threads for j = 1:samples
#         p, v = sample_probability(Γ)
#         w_means[j] = p * Int(sum(v))
#         ps[j] = p
#     end
#     ws[i] = sum(w_means) / sum(ps)
# end

# p = plot(βs, ws, label="N = 100, J = 1")
# xlabel!("β")
# ylabel!("w_crit")
# display(p)

##############

N = 20
J = 1
β = 10
α = 1

samples = 100
progress = Progress(samples)

ps_avg = zeros(N + 1)
counts_avg = zeros(N + 1)

for i = 1:samples
    Γ = Gaussian.rand_covariance(N, β, J)
    ps, counts = Gaussian.full_sample(Γ, α)
    global ps_avg += ps
    global counts_avg += counts
    update!(progress, i)
end

ps_avg ./= samples
counts_avg ./= samples

p = bar(0:2:N, log2.(ps_avg[1:2:N+1] ./ counts_avg[1:2:N+1]))
xlabel!("w")
ylabel!("log2(avg prob(w))")
display(p)
