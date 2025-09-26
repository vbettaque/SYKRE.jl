module Gaussian

using LinearAlgebra
using SkewLinearAlgebra
using Distributions
using Base.Threads
using ProgressMeter


function bitvec(n, len::Integer)
    @assert n >= 0
    @assert len > 0

    vec = zeros(Integer, len)
    return digits!(vec, n, base=2)
end

function indexed_even_bitvec(i, len::Integer)
    @assert len > 1
    @assert 1 <= i <= (big"2")^(len - 1)

    n = i - 1
    vec = zeros(len)
    vec[1] = isodd(count_ones(n))
    vec[2:len] = bitvec(n, len - 1)
    return vec
end

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


function full_sample(Γ, α=1)
    N, N_ = size(Γ)
    Γ_ = I + Matrix(Γ)
    purity = det(Γ_)

    counts = zeros(N + 1)
    ps = zeros(N + 1)

    ps[1] = (1 / purity)^α
    counts[1] = 1

    Threads.@threads for i = 2:2^(N - 1)
        v =  indexed_even_bitvec(i, N)
        v_bit = isone.(v)
        w = Int(sum(v))
        ps[w + 1] += (det(Γ[v_bit, v_bit]) / purity)^α
        counts[w + 1] += 1
    end

    return ps, counts
end

end
