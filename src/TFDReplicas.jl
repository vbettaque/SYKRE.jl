module TFDReplicas

using LinearAlgebra

using ..SYK
using ..SYKMatrix
using ..Replicas


function propagtors(Σ::ReplicaMatrix, syk::SYKData)
    Δτ = syk.β / Σ.L

    D = Replicas.differentials(Σ.R, Σ.L; periodic = false)
    D2 = Replicas.differentials(Σ.R, Σ.L; periodic = true)
    D2.blocks[Σ.L÷2+1, Σ.L÷2, 1] *= -1

    prop = D - Δτ^2 * Σ
	prop2 = D2 - Δτ^2 * Σ

    return prop, prop2
end


function pfaffians(Σ::ReplicaMatrix, syk::SYKData)
    prop, prop2 = propagtors(Σ, syk)
    det = Replicas.det(prop_minus)
    det2 = Replicas.det(prop_plus)
    return sqrt(det), sqrt(det2)
end


function G_SD(Σ::ReplicaMatrix, w::Real, syk::SYKData)
    prop, prop2 = propagtors(Σ, syk)
    G = if iszero(w)
        -inv(prop)'
    elseif isone(w)
        -inv(prop2)'
    else
        -(w * inv(prop) + (1 - w) * inv(prop2))'
    end
    return G
end


function Σ_SD(G::ReplicaMatrix, syk::SYKData)
	return syk.J^2 * G.^(syk.q - 1)
end


function schwinger_dyson(G_init::ReplicaMatrix, w, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    @assert iseven(syk.q)
    @assert 0 ≤ w ≤ 1

	t = init_lerp

    G = G_init
	Σ = Σ_SD(G, syk)

    i = 1

    G_new = G_SD(Σ, w, syk)
    G_lerp = Replicas.init(G_init.R, G_init.L)

    err = frobenius(G_new - G) / frobenius(G)

    @info "Iteration $(i)" rel_error = err lerp = t

	while i <= max_iters
		G_lerp = t * G_new + (1 - t) * G

		err_new = frobenius(G_lerp - G) / frobenius(G)

		if err_new < tol
            G = G_lerp
            Σ = Σ_SD(G, syk)
            @info "Converged after $(i) iterations"
            break
        end

		if (err_new > err)
            @info "Relative error increased to $(err_new), trying again..."
            t /= lerp_divisor
            continue
        end

        err = err_new

		G = G_lerp
        Σ = Σ_SD(G, syk)

        i += 1

        if i > max_iters
            @warn "Exceeded iterations!"
            break
        end

        @info "Iteration $(i)" rel_error = err lerp = t
        G_new = G_SD(Σ, w, syk)
	end

	return G, Σ
end


function action(G::ReplicaMatrix, Σ::ReplicaMatrix, w, syk::SYKData)
    Δτ = syk.β / G.L

    pf, pf2 = pfaffians(Σ, syk)
    @debug pf_minus, pf2

    prop_term = if iszero(w)
        -log(pf)
    elseif isone(w)
        -log(pf2)
    else
        -(w * log(pf) + (1 - w) * log(pf2))
    end
    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * Δτ^2 * sum(x -> x^syk.q, G)
    @debug prop_term, on_shell_term

    action = syk.N * (prop_term + on_shell_term)
    @debug action

    return action
end


log_saddle(G::ReplicaMatrix, Σ::ReplicaMatrix, w, syk::SYKData) = -action(G, Σ, w, syk)


log2_saddle(G::ReplicaMatrix, Σ::ReplicaMatrix, w, syk::SYKData) = log(2, ℯ) * log_saddle(G, Σ, w, syk)


end
