module Gaussian

using LinearAlgebra
using SkewLinearAlgebra
using Distributions
using Base.Threads
using Plots


function rand_hamiltonian(N, J)
    @assert iseven(N)

    std = J / sqrt(N)
    normal = Normal(0, std)
    H = rand(normal, N, N)
    return skewhermitian!(H)
end

function rand_covariance(N, β, J)
    @assert iseven(N)

    H = rand_hamiltonian(N, J)
    return tanh(β * H)
end

function probability(Γ::AbstractMatrix, v::AbstractVector)
    @assert all(isone.(v) .|| iszero.(v))

    entries = isone.(v)
    sub_matrix = Γ[entries, entries]
    return det(sub_matrix) / det(I + Matrix(Γ))
end

function sample_probability(Γ)
    N, N_ = size(Γ)
    Γ_ = I + Matrix(Γ)
    purity = det(Γ_)
    prob = 1
    v = ones(Bool, N)
    for i = 1:N
        @views Γ_[i, i] -= 1
        v[i] = false
        p = det(Γ_[v, v]) / purity
        cond_prob = p / prob
        if rand() >= cond_prob
            v[i] = true
        end
        prob *= cond_prob
    end
    return prob, v
end



N = 200
J = 1
samples = 200

βs = collect(1:300) / 100
ws = zeros(length(βs))

H = rand_hamiltonian(N, J)

for i = eachindex(βs)
    β = βs[i]
    println("β = ", β)
    Γ = tanh(β * H)
    w_means = zeros(samples)
    ps = zeros(samples)
    Threads.@threads for j = 1:samples
        p, v = sample_probability(Γ)
        w_means[j] = p * Int(sum(v))
        ps[j] = p
    end
    ws[i] = sum(w_means) / sum(ps)
end

p = plot(βs, ws, label="N = 200, J = 1")
xlabel!("β")
ylabel!("w_crit")
display(p)

end
