module WeightedReplicas

using LinearAlgebra

using ..SYK
using ..SYKMatrix
using ..Replicas


function propagtors(Σ::ReplicaMatrix, syk::SYKData)
    Δτ = syk.β / Σ.L

    D_minus = Replicas.differentials(Σ.M, Σ.L; periodic = false)
    D_plus = Replicas.differentials(Σ.M, Σ.L; periodic = true)

    prop_minus = D_minus - Δτ^2 * Σ
	prop_plus = D_plus - Δτ^2 * Σ

    return prop_minus, prop_plus
end


function pfaffians(Σ::ReplicaMatrix, syk::SYKData)
    prop_minus, prop_plus = propagtors(Σ, syk)
    det_minus = Replicas.det(prop_minus)
    det_plus = Replicas.det(prop_plus)
    if det_minus < 0 && syk.M == 1
        det_minus = abs(det_minus)
    end
    if det_plus < 0 && syk.M == 1
        det_plus = abs(det_plus)
    end
    return sqrt(det_minus), sqrt(det_plus)
end


function G_SD(Σ::ReplicaMatrix, w::Real, syk::SYKData)
    prop_minus, prop_plus = propagtors(Σ, syk)
    G = if iszero(w)
        -inv(prop_minus)'
    elseif isone(w)
        -inv(prop_plus)'
    else
        -(w * inv(prop_plus) + (1 - w) * inv(prop_minus))'
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
    G_lerp = Replicas.init(G_init.M, G_init.L)

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

    # plot_matrix(G; title="w = $(w), β = $(syk.β)")

	return G, Σ
end


function action(G::ReplicaMatrix, Σ::ReplicaMatrix, w, syk::SYKData)
    Δτ = syk.β / G.L

    pf_minus, pf_plus = pfaffians(Σ, syk)
    @debug pf_minus, pf_plus

    prop_term = if iszero(w)
        -log(pf_minus)
    elseif isone(w)
        -log(pf_plus)
    else
        -(w * log(pf_plus) + (1 - w) * log(pf_minus))
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
