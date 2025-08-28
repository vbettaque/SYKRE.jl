module WeightReplicas

using BlockArrays
using Plots

using ..SYK
using ..SYKMatrix
using ..Replicas


function plot_matrix(A::ReplicaMatrix; title="")
    A_M = convert(Matrix{Float64}, A)
    p = heatmap(A_M, aspect_ratio = 1, clims=(-0.5, 0.5), yflip = true, color = :greys, title=title)

    display(p)
end

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
    return sqrt(det(prop_minus)), sqrt(det(prop_plus))
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
    @assert iseven(syk.M) && syk.M == G_init.M
    @assert iseven(syk.q)
    @assert 0 ≤ w ≤ 1


	t = init_lerp

    G = G_init
	Σ = Σ_SD(G, syk)

    i = 1
    println("Iteration ", i)
    G_new = G_SD(Σ, w, syk)

    err = frobenius(G_new - G) / frobenius(G)

	while i <= max_iters
		G_lerp = t * G_new + (1 - t) * G

		err_new = frobenius(G_lerp - G) / frobenius(G)
        println("err_new = ", err_new)
		if err_new < tol
            G = G_lerp
            Σ = Σ_SD(G, syk)
            println("Converged after ", i, " iterations")
            break
        end

		if (err_new > err)
            println("Relative error increased, trying again...")
            t /= lerp_divisor
            continue
        end

        err = err_new
        println("err = ", err, ", t = ", t)

		G = G_lerp
        Σ = Σ_SD(G, syk)

        i += 1

        if i > max_iters
            println("Exceeded iterations!")
            break
        end

        println("Iteration ", i)
        G_new = G_SD(Σ, w, syk)
	end

    plot_matrix(G; title="w = $(w), β = $(syk.β)")

	return G, Σ
end


function action(G::ReplicaMatrix, Σ::ReplicaMatrix, w, syk::SYKData)
    Δτ = syk.β / G.L

    pfaff_minus, pfaff_plus = pfaffians(Σ, syk)
    prop_term = if iszero(w)
        -log(pfaff_minus)
    elseif isone(w)
        -log(pfaff_plus)
    else
        -(w * log(pfaff_plus) + (1 - w) * log(pfaff_minus))
    end
    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * Δτ^2 * sum(x -> x^syk.q, G)

    return syk.N * (prop_term + on_shell_term)
end


log_trace_swap(G::ReplicaMatrix, Σ::ReplicaMatrix, w, syk::SYKData) = -action(G, Σ, w, syk)


log2_trace_swap(::ReplicaMatrix, Σ::ReplicaMatrix, w, syk::SYKData) = log(2, ℯ) * log_trace_swap(G, Σ, w, syk)


end
