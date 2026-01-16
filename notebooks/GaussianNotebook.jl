using CSV, DataFrames, Statistics, CairoMakie, LinearAlgebra, FFTW, Latexify, ProgressMeter

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
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

N = 50
J = 1
β = 50
α = 2

samples = 1000000
progress = Progress(samples)

Γ = Gaussian.rand_covariance(N, β, J)

purity = det(Γ)

ps_avg = zeros(N + 1)
counts = zeros(N + 1)

ps_avg[1] = (1 / purity)^α
counts[1] = 1

for i = 1:samples
    p, v = Gaussian.sample_probability(Γ)
    w = Int(sum(v))
    iszero(w) && @info("Identity!")
    global ps_avg[w+1] += p^α
    global counts[w+1] += 1
    ProgressMeter.update!(progress, i)
end

ps_avg ./= counts

p = barplot(0:2:N, log.(binomial.(N, 0:2:N) .* ps_avg[1:2:N+1]) / N)
