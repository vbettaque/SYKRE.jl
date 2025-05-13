module SYK

using Integrals

export SYKData

struct SYKData
    N::Int
    J::Float64
    q::Int
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

end
