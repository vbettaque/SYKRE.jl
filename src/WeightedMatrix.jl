module WeightedMatrix

using LinearAlgebra
using BlockArrays
using Plots
using Colors

using ..SYK
using ..SYKMatrix


function plot_matrix(M; title="")
    blue = RGB(0,101.0/255,1)
    orange = RGB(1,154.0/255,0)
    grad = cgrad([blue, :gray95, orange], [0.0, 0.5, 1.0])
    p = heatmap(M, aspect_ratio = 1, clims=(-1, 1), yflip = true, color = grad, title=title)

    display(p)
end


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


function differential(L; M = 1, periodic=false)
    D = Matrix{Float64}(I, L, L)
    for i=1:L-1
        D[i+1, i] = -1
    end
    D[1, L] = periodic ? -1 : 1
    I_M = Matrix{Float64}(I, M, M)
	return blockkron(I_M, D)
end


function propagtors(Σ, syk::SYKData)
    M, L, Δτ = block_structure(Σ, syk)

    D_minus = differential(L; M = M, periodic = false)
    D_plus = differential(L; M = M, periodic = true)

    prop_minus = D_minus - 2 * Δτ^2 * Σ
	prop_plus = D_plus - 2 * Δτ^2 * Σ

    return prop_minus, prop_plus
end


function pfaffians(Σ, syk::SYKData)
    prop_minus, prop_plus = propagtors(Σ, syk)
    return sqrt(det(prop_minus)), sqrt(det(prop_plus))
end


function G_SD(Σ, w, syk::SYKData)
    prop_minus, prop_plus = propagtors(Σ, syk)
    G = if iszero(w)
        -2 * inv(prop_minus)'
    elseif isone(w)
        -2 * inv(prop_plus)'
    else
        -2 * (w * inv(prop_plus) + (1 - w) * inv(prop_minus))'
    end
    # G -= Diagonal(G)
    return G
end


function Σ_SD(G, syk::SYKData)
	return syk.J^2 * G.^(syk.q - 1)
end


function schwinger_dyson(G_init, w, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
    @assert iseven(syk.M)
    @assert iseven(syk.q)
    @assert 0 ≤ w ≤ 1

    plot_matrix(G_init; title="G_init")

	t = init_lerp

    G = G_init
	Σ = Σ_SD(G, syk)

    i = 1
    println("Iteration ", i)
    G_new = G_SD(Σ, w, syk)

    err = sum(abs.(G_new - G)) / sum(abs.(G))

	while i <= max_iters
		G_lerp = t * G_new + (1 - t) * G

		err_new = sum(abs.(G_lerp - G)) / sum(abs.(G))
		if isapprox(err_new, 0; atol=1e-5)
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


function action(G, Σ, w, syk::SYKData)
    L, M, Δτ = block_structure(G, syk)

    pfaff_minus, pfaff_plus = pfaffians(Σ, syk)
    prop_term = if iszero(w)
        -log(pfaff_minus)
    elseif isone(w)
        -log(pfaff_plus)
    else
        -(w * log(pfaff_plus) + (1 - w) * log(pfaff_minus))
    end
    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * Δτ^2 * sum(G.^syk.q)

    return syk.N * (prop_term + on_shell_term)
end


log_saddle(G, Σ, w, syk) = -action(G, Σ, w, syk)


log2_saddle(G, Σ, w, syk::SYKData) = log(2, ℯ) * log_saddle(G, Σ, w, syk)


end
