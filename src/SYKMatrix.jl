module SYKMatrix

using SparseArrays
using LinearAlgebra

using ..SYK

# function differential(L)
# 	rows = repeat(1:L, inner=2)
# 	values = repeat([-1, 1], L)
# 	columns = rows + values; columns[1] = L; columns[2*L] = 1
# 	values[1] *= -1; values[2*L] *= -1
# 	return SkewHermitian(Matrix(sparse(rows, columns, values)))
# end

function differential(L)
	rows = repeat(1:L, inner=2)
	values = repeat([-1., 1.], L)
	columns = rows + repeat([-1, 0], L); columns[1] = L
	values[1] *= -1
	return Matrix(sparse(rows, columns, values))
end


function G_SD(Σ, syk::SYKData)
	L, _ = size(Σ)
    Δτ = syk.β/L
	D = differential(L)
    prop = D - Δτ^2 * Σ
    return inv(prop)
end


function Σ_SD(G, syk::SYKData)
	return syk.J^2 * G.^(syk.q-1)
end


function schwinger_dyson(L, syk::SYKData; Σ_init = zeros(L, L), max_iters=1000)
	t = 0.5; b = 2; err=0
	Σ = Σ_init
	G = G_SD(Σ, syk)
	for i=1:max_iters
		Σ = Σ_SD(G, syk)
		G_new = t * G_SD(Σ, syk) + (1 - t) * G
		err_new = sum(abs.(G_new - G)) #/ sum(abs.(G))
		if isapprox(err_new, 0; atol=1e-10)
            G = G_new
            Σ = Σ_SD(G, syk)
            println("Converged after ", i, " iterations")
            break
        end
        if err_new > err && i > 1
            t /= b
            i -= 1
            continue
        end
		err = err_new
		G = G_new
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ, G
end


function action(Σ, G, syk::SYKData)
    L, _ = size(Σ)
    Δτ = syk.β/L
    D = differential(L)
    prop_term = -log(det(D - Δτ^2 * Σ))/2
    greens_term = -1/2 * syk.J^2/syk.q * Δτ^2 * sum(G.^syk.q)
    lagrange_term = 1/2 * Δτ^2 * tr(G * transpose(Σ))
    return syk.N * (prop_term + greens_term + lagrange_term)
end


function logZ(L, syk::SYKData; Σ_init = zeros(L, L), max_iters=1000)
    Σ, G = schwinger_dyson(L, syk; Σ_init = Σ_init, max_iters=max_iters)
    return -action(Σ, G, syk), Σ
end


function free_energy(L, syk::SYKData; Σ_init = zeros(L, L), max_iters=1000)
    logZ_saddle, Σ = logZ(L, syk; Σ_init=Σ_init, max_iters=max_iters)
    return - logZ_saddle / syk.β, Σ
end


function log_purity(L, syk::SYKData; Σ_init = zeros(L, L), max_iters=1000)
    syk_2β = SYKData(syk.N, syk.J, syk.q, syk.M, 2*syk.β)
    Σ, G = schwinger_dyson(L, syk_2β; Σ_init = Σ_init, max_iters=max_iters)
    return -action(Σ, G, syk), Σ
end


function renyi2(L_β, syk_β::SYKData; Σ_β_init = zeros(L_β, L_β), Σ_2β_init = zeros(4L_β, 4L_β), max_iters=1000)
    L_2β = 4 * L_β
    syk_2β = SYKData(syk_β.N, syk_β.J, syk_β.q, syk_β.M, 2*syk_β.β)
    Σ_β, G_β = schwinger_dyson(L_β, syk_β; Σ_init = Σ_β_init, max_iters=max_iters)
    Σ_2β, G_2β = schwinger_dyson(L_2β, syk_2β; Σ_init = Σ_2β_init, max_iters=max_iters)

    logZ_β_saddle = - syk_β.N * log(2, ℯ) * action(Σ_β, G_β, syk_β)
    logZ_2β_saddle = - syk_β.N * log(2, ℯ) * action(Σ_2β, G_2β, syk_2β)

    return -(logZ_2β_saddle - 2 * logZ_β_saddle), Σ_β, Σ_2β
end


end
