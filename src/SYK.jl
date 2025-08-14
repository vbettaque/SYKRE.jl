module SYK

using Integrals

export SYKData

struct SYKData
    N::Int
    J::Float64
    q::Float64
    M::Int
    β::Float64
end

function free_energy_q2(β, J; N=1)
    f(θ, p) = cos(θ)^2 / π * log(1 + exp(-2 * β * J * sin(θ)))
    domain = (0, π)
    prob = IntegralProblem(f, domain)
    logZ = N * solve(prob, QuadGKJL(); reltol = 1e-3, abstol = 1e-3).u
    E_0 = N * -2J/(3π)
    return E_0 - logZ / β
end

function zero_temperature_entropy(q; N=1)
    S_0 = N * log(2)/2
    f(x, p) = π * (1/2 - x) * tan(π * x)
    domain = (0, 1/q)
    prob = IntegralProblem(f, domain)
    S_0 -= N * solve(prob, QuadGKJL(); reltol = 1e-3, abstol = 1e-3).u
    return S_0
end

function high_temp_free_energy(β, q, J; N=1)
    J_q = J * sqrt(q) / 2^((q-1)/2)
    x = β * J_q^2
    υ = x - x^3/8 + 65 * x^5/(16 * factorial(5)) - 3787 * x^7 / (64 * factorial(7))
    logZ = log(2)/2
    logZ += υ / q^2 * (tan(υ / 2) - υ / 4)
    logZ += υ / q^3 * (υ - 2 * tan(υ / 2) * (1 - υ^2 / 12))
    return -N * logZ / β
end


end
