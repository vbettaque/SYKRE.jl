module MatrixSD

using SparseArrays
using LinearAlgebra
using SkewLinearAlgebra

function differential(L; anti_periodic=true)
	rows = repeat(1:L, inner=2)
	values = repeat([-1, 1], L)
	columns = rows + values; columns[1] = L; columns[2*L] = 1
	anti_periodic && (values[1] *= -1; values[2*L] *= -1)
	return SkewHermitian(sparse(rows, columns, float.(values)))
end

function G_eom(Σ::SkewHermitian, β, M; sre=false)
	L, _ = size(Σ)
    Δτ = β/L
	D_minus = differential(L)
    @assert isskewsymmetric(Σ) "Not skew-symmetric!"
    prop_minus = D_minus + Δτ^2 * Σ
    sre || return inv(prop_minus)

	D_plus = differential(L, anti_periodic=false)
	prop_plus = D_plus + Δτ^2 * Σ
	pfaff_minus = pfaffian(prop_minus)^M
	pfaff_plus = pfaffian(prop_plus)^M
	p_minus = pfaff_minus / (pfaff_minus + pfaff_plus)
    isapprox(p_minus, 1) && return inv(prop_minus)
    isapprox(p_minus, 0) && return inv(prop_plus)
	return p_minus * inv(prop_minus) + (1 - p_minus) * inv(prop_plus)
end

function Σ_eom(G::SkewHermitian, q, J)
	return SkewHermitian(J^2 * map(g -> g^(q-1), G))
end

function action(Σ::SkewHermitian, G::SkewHermitian, β, M, q, J; sre=false)
    L, _ = size(Σ)
    Δτ = β/L
    D_minus = differential(L)
    prop_minus = D_minus + Δτ^2 * Σ
    prop_term = 0
    if sre
        D_plus = differential(L, anti_periodic=false)
	    prop_plus = D_plus + Δτ^2 * Σ
        prop_term = -log(pfaffian(prop_minus)^M + pfaffian(prop_plus)^M)
    else
        prop_term = -M * log(pfaffian(prop_minus))
    end
    # println("prop = ", prop_term)
    greens_term = -Δτ^2 * M/2 * J^2/q * sum(map(g -> g^q, G))
    # println("greens = ", greens_term)
    lagrange_term = Δτ^2 * M/2 * tr(G * transpose(Σ))
    # println("lagrange = ", lagrange_term)
    return prop_term + greens_term + lagrange_term
end

function schwinger_dyson(L, β, M, q, J; sre=false, max_iters=10000)
	x = 0.5; b = 2; err=0
	Σ = SkewHermitian(zeros(L, L))
	G = G_eom(Σ, β, M, sre=sre)
	for i=1:max_iters
		Σ = Σ_eom(G, q, J)
		G_new = x * G_eom(Σ, β, M, sre=sre) + (1 - x) * G
		err_new = sum(abs.(G_new - G)) / sum(abs.(G))
		isapprox(err_new, 0) && break
		err_new > err && (x /= b)
		err = err_new
		G = G_new
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ, G
end

 # Add loop corrections?
function logZ_saddlepoint(L, β, M, q, N, J; max_iters=10000)
    Σ, G = schwinger_dyson(L, β, M, q, J, max_iters=max_iters)
    return -N * action(Σ, G, β, M, q, J)
end

function renyi2_saddlepoint(L, β, q, N, J; max_iters=10000)
    Σ, G = schwinger_dyson(L, β, 2, q, J, sre=true, max_iters=max_iters)
    return N * log(2, ℯ) * action(Σ, G, β, 2, q, J, sre=true)
end

 # Add loop corrections?
function sre_saddlepoint(L, β, α, q, N, J; max_iters=10000)
    M = 2 * α
    Σ, G = schwinger_dyson(L, β, M, q, J, sre=true, max_iters=max_iters)
    sre = -N * log(2, ℯ) * action(Σ, G, β, M, q, J, sre=true)
    renyi2 = renyi2_saddlepoint(L, β, q, N, J; max_iters=max_iters)
    return (sre + α * renyi2 - 0.5) / (1 - α)
end

end