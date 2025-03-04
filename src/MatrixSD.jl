module MatrixSD

using SparseArrays
using LinearAlgebra

function differential(L, anti_periodic=true)
	rows = repeat(1:L, inner=2)
	values = repeat([-1, 1], L)
	columns = rows + values; columns[1] = L; columns[2*L] = 1
	anti_periodic && (values[1] *= -1; values[2*L] *= -1)
	return sparse(rows, columns, float.(values))
end

function G(Σ, Δτ, α)
	L, _ = size(Σ)
	D_minus = differential(L)
	D_plus = differential(L)
	prop_minus = D_minus + Δτ^2 * Σ
	prop_plus = D_plus + Δτ^2 * Σ
	pfaff_minus = det(prop_minus)^α
	pfaff_plus = det(prop_plus)^α
	p_minus = pfaff_minus / (pfaff_minus + pfaff_plus)
	return p_minus * inv(prop_minus) + (1 - p_minus) * inv(prop_plus)
end

function Σ(G, J, q)
	return J^2 * map(g -> g^(q-1), G)
end

function sre_action(Σ, G, β, N, q, J, α)
    L, _ = size(Σ)
    Δτ = β/L
    D_minus = differential(L)
	D_plus = differential(L)
    prop_minus = D_minus + Δτ^2 * Σ
	prop_plus = D_plus + Δτ^2 * Σ
    prop_term = -log(det(prop_minus)^α + det(prop_plus)^α)
    println("prop = ", prop_term)
    greens_term = - Δτ^2 * α  * J^2/q * sum(map(g -> g^q, G))
    println("greens = ", greens_term)
    lagrange_term = Δτ^2 * α * tr(G * transpose(Σ))
    println("lagrange = ", lagrange_term)
    return N * (prop_term + greens_term + lagrange_term)
end

function sre_saddlepoint(Σ, G, β, N, q, J, α)
    return -sre_action(Σ, G, β, N, q, J, α) / (1 - α) - N # Add loop corrections?
end

function sre_schwinger_dyson(q, J, β, α, L; max_iters=10000)
	Δτ = β/L
	x = 0.5; b = 2; err=0
	Σ_ = zeros(L, L)
	G_ = G(Σ_, Δτ, α)
	for i=1:max_iters
		Σ_ = Σ(G_, J, q)
		G_new = x * G(Σ_, Δτ, α) + (1 - x) * G_
		err_new = sum(abs.(G_new - G_)) / sum(abs.(G_))
		isapprox(err_new, 0; atol=eps()) && break
		err_new > err && (x /= b)
		err = err_new
		G_ = G(Σ_, Δτ, α)
	end
	return Σ_, G_
end

# ╔═╡ 42e9b126-8c2f-4786-9a12-53c9f8d7de7d
sre_schwinger_dyson(4, 1, 10, 2, 1000)


end