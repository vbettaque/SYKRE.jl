module SYKMatrix

using LinearAlgebra
using BlockArrays
using Plots

using ..SYK


function block_structure(A, syk::SYKData)
    Ls = blocksizes(A)
    @assert allequal(Ls)
    L, L_ = Ls[1, 1]
    @assert L == L_
    M, M_ = size(Ls)
    @assert M == M_ == syk.M
    Δτ = syk.β / L
    return M, L, Δτ
end


function differential(L; M = 1)
    D = Matrix{Float64}(I, L, L)
    for i=1:L-1
        D[i+1, i] = -1
    end
    D[1, L] = 1
    I_M = Matrix{Float64}(I, M, M)
	return blockkron(I_M, D)
end


function propagtor(Σ, syk::SYKData)
    M, L, Δτ = block_structure(Σ, syk)
    D = differential(L; M = M)
    return D - Δτ^2 * Σ
end


function G_SD(Σ, syk::SYKData)
	prop = propagtor(Σ, syk)
    return -inv(prop)'
end


function Σ_SD(G, syk::SYKData)
	return syk.J^2 * G.^(syk.q-1)
end


function schwinger_dyson(G_init, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
    @assert iseven(syk.q)

	t = init_lerp; err=0

    G = G_init
	Σ = Σ_SD(G, syk)

    i = 1
    println("Iteration ", i)
    G_new = G_SD(Σ, syk)

	while i <= max_iters
		G_lerp = t * G_new + (1 - t) * G

		err_new = sum(abs.(G_lerp - G)) / sum(abs.(G))
		if isapprox(err_new, 0; atol=1e-10)
            G = G_lerp
            Σ = Σ_SD(G, syk)
            println("Converged after ", i, " iterations")
            break
        end

		if (err_new > err && i > 1)
            println("Relative error increased, trying again...")
            t /= lerp_divisor
            continue
        end

        err = err_new
        println("err = ", err, ", t = ", t)      

		G = G_lerp
        Σ = Σ_SD(G, syk)

        # p = plot(Gray.(G .- minimum(G)), title="Iteration $(i)")
        # display(p)

        i += 1

        if i > max_iters
            println("Exceeded iterations!")
            break
        end

        println("Iteration ", i)
        G_new = G_SD(Σ, syk)
	end
	return G, Σ
end


function action(G, Σ, syk::SYKData)
    L, M, Δτ = block_structure(G, syk)

    prop = propagtor(Σ, syk)
    prop_term = -log(det(prop))/2
    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * Δτ^2 * sum(G.^syk.q)
    return syk.N * (prop_term + on_shell_term)
end


logZ_saddle(G, Σ, syk::SYKData) = -action(G, Σ, syk)

log2Z_saddle(G, Σ, syk::SYKData) = log(2, ℯ) * logZ_saddle(G, Σ, syk)

free_energy_saddle(G, Σ, syk::SYKData) = -logZ_saddle(G, Σ, syk) / syk.β


function log_purity_saddle(G_init, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
    syk_2β = SYKData(syk.N, syk.J, syk.q, syk.M, 2*syk.β)
    G, Σ = schwinger_dyson(G_init, syk_2β; init_lerp = init_lerp, lerp_divisor = lerp_divisor, max_iters = max_iters)
    return logZ_saddle(G, Σ, syk_2β)
end

function log2_purity_saddle(G_init, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
    return log(2, ℯ) * log_purity_saddle(G_init, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, max_iters = max_iters)
end

function renyi2_saddle(G_β_init, G_2β_init, syk_β::SYKData; params_β = (0.5, 2, 1000), params_2β = (0.5, 2, 1000))
    L_β, _, _ = block_structure(G_β_init, syk)
    L_2β, _, _ = block_structure(G_2β_init, syk)
    @assert L_2β = 2 * L_β

    syk_2β = SYKData(syk_β.N, syk_β.J, syk_β.q, syk_β.M, 2*syk_β.β)
    G_β, Σ_β = schwinger_dyson(G_β_init, syk_β; init_lerp, lerp_divisor, max_iters = params_β)
    G_2β, Σ_2β = schwinger_dyson(G_2β_init, syk_2β; init_lerp, lerp_divisor, max_iters = params_2β)

    log2Z_β_saddle = log2Z_saddle(G_β, Σ_β, syk)
    log2Z_2β_saddle = log2Z_saddle(G_2β, Σ_2β, syk)

    return -(log2Z_2β_saddle - 2 * log2Z_β_saddle)
end

end