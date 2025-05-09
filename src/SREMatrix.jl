module SREMatrix

using SparseArrays
using LinearAlgebra
using SkewLinearAlgebra

using ..SYK

function differential(L; anti_periodic=true)
	rows = repeat(1:L, inner=2)
	values = repeat([-1, 1], L)
	columns = rows + values; columns[1] = L; columns[2*L] = 1
	anti_periodic && (values[1] *= -1; values[2*L] *= -1)
	return SkewHermitian(Matrix(sparse(rows, columns, float.(values))))
end

function G_SD(Σ::SkewHermitian, syk::SYKData)
    @assert iseven(syk.M)
	L, _ = size(Σ)
    Δτ = syk.β/L
	D_minus = differential(L)
    D_plus = differential(L, anti_periodic=false)
    prop_minus = D_minus - Δτ^2 * Σ
	prop_plus = D_plus - Δτ^2 * Σ
	pfaff_minus = pfaffian(prop_minus)^syk.M
	pfaff_plus = pfaffian(prop_plus)^syk.M
    # println("pfaff_plus = ", pfaff_plus)
    # println("pfaff_minus = ", pfaff_minus)
	p_plus = pfaff_plus / (pfaff_minus + pfaff_plus)
    # println("p_plus = ", p_plus)
    # isapprox(p_plus, 0; atol=1e-50) && return inv(prop_minus)
    # isapprox(p_plus, 1; atol=1e-50) && return inv(prop_plus)
	return (1 - p_plus) * inv(prop_minus) + p_plus * inv(prop_plus)
end

function Σ_SD(G::SkewHermitian, syk::SYKData)
	return SkewHermitian(syk.J^2 * map(g -> g^(syk.q-1), G))
end

# function action(Σ::SkewHermitian, G::SkewHermitian, β, M, q, J; sre=false)
#     L, _ = size(Σ)
#     Δτ = β/L
#     D_minus = differential(L)
#     prop_minus = D_minus - Δτ^2 * Σ
#     prop_term = 0
#     if sre
#         D_plus = differential(L, anti_periodic=false)
# 	    prop_plus = D_plus - Δτ^2 * Σ
#         prop_term = -log(pfaffian(prop_minus)^M + pfaffian(prop_plus)^M)
#     else
#         prop_term = -M * log(pfaffian(prop_minus))
#     end
#     # println("prop = ", prop_term)
#     greens_term = -Δτ^2 * M/2 * J^2/q * sum(map(g -> g^q, G))
#     # println("greens = ", greens_term)
#     lagrange_term = Δτ^2 * M/2 * tr(G * transpose(Σ))
#     # println("lagrange = ", lagrange_term)
#     return prop_term + greens_term + lagrange_term
# end

function schwinger_dyson(L, syk::SYKData; Σ_init = SkewHermitian(zeros(L, L)), max_iters=10000)
	t = 0.5; b = 2; err=0
	Σ = Σ_init
	G = G_SD(Σ, syk)
	for i=1:max_iters
		Σ = Σ_SD(G, syk)
		G_new = t * G_SD(Σ, syk) + (1 - t) * G
		err_new = sum(abs.(G_new - G)) / sum(abs.(G))
		isapprox(err_new, 0) && break
		err_new > err && (t /= b)
		err = err_new
		G = G_new
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ, G
end


# function sre_saddlepoint(L, βs, α, q, N, J; Σ_init = SkewHermitian(zeros(L, L)), max_iters=10000)
#     steps = length(βs)
#     M = 2 * α
#     sres = zeros(steps)
#     Σ_α = Σ_init
#     Σ_2 = Σ_init
#     Σ_Z = Σ_init
#     for i=1:steps
#         println(i, " of ", steps)

#         Σ_α, G_α = schwinger_dyson(L, βs[i], M, q, J, Σ_init=Σ_α, sre=true, max_iters=max_iters)
#         Σ_2, G_2 = schwinger_dyson(L, βs[i], 2, q, J, Σ_init=Σ_2, sre=true, max_iters=max_iters)
#         Σ_Z, G_Z = schwinger_dyson(L, βs[i], 1, q, J, Σ_init=Σ_Z, max_iters=max_iters)

#         sre_α_saddle = -N * log(2, ℯ) * action(Σ_α, G_α, βs[i], M, q, J, sre=true)
#         sre_2_saddle = -N * log(2, ℯ) * action(Σ_2, G_2, βs[i], 2, q, J, sre=true)
#         logZ_saddle = -N * log(2, ℯ) * action(Σ_Z, G_Z, βs[i], 1, q, J)

#         sres[i] = (sre_α_saddle - sre_2_saddle) / (1 - α) + 2 * logZ_saddle
#         println("(T, SRE) = (", 1/βs[i], ", ", sres[i], ")")
#     end

#     return sres
# end


#  # Add loop corrections?
# function logZ_saddlepoint(L, β, M, q, N, J; Σ_init = SkewHermitian(zeros(L, L)), max_iters=10000)
#     Σ, G = schwinger_dyson(L, β, M, q, J, Σ_init=Σ_init, max_iters=max_iters)
#     return -N * log(2, ℯ) * action(Σ, G, β, M, q, J)
# end

# function renyi2_saddlepoint(L, β, q, N, J; Σ_init = SkewHermitian(zeros(L, L)), max_iters=10000)
#     Σ, G = schwinger_dyson(L, β, 2, q, J, Σ_init=Σ_init, sre=true, max_iters=max_iters)
#     return -N * log(2, ℯ) * action(Σ, G, β, 2, q, J, sre=true)
# end

#  # Add loop corrections?
# function sre_saddlepoint(L, β, α, q, N, J; Σ_init = SkewHermitian(zeros(L, L)), max_iters=10000)
#     M = 2 * α
#     Σ, G = schwinger_dyson(L, β, M, q, J, Σ_init=Σ_init, sre=true, max_iters=max_iters)
#     sre = -N * log(2, ℯ) * action(Σ, G, β, M, q, J, sre=true)
#     # println("sre = ", sre)
#     renyi2 = renyi2_saddlepoint(L, β, q, N, J; Σ_init=Σ_init, max_iters=max_iters)
#     # println("renyi = ", renyi2)
#     return (sre - renyi2) / (1 - α) + 2 * logZ_saddlepoint(L, β, 1, q, N, J; Σ_init=Σ_init, max_iters=max_iters)
# end

end
