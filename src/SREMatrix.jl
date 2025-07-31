module SREMatrix

using LinearAlgebra
using BlockArrays
using Plots

using ..SYK
using ..SYKMatrix

# TODO
# function time_invariance(M, β)
#     L, _ = size(M)
#     Δτ = β / L
#     τs = LinRange(0, 2β - Δτ, 2 * L)
#     M_invariant = zeros(2 * L)
#     M_invariant[1:L] = M[1:L, 1]
#     M_invariant[L+2:2L] = reverse(M[1, 2:L])
#     M_invariant[L+1] = (M_invariant[L] + M_invariant[L+2])/2
#     return τs, M_invariant
# end


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

    prop_minus = D_minus - Δτ^2 * Σ
	prop_plus = D_plus - Δτ^2 * Σ

    return prop_minus, prop_plus
end


function pfaffians(G, syk::SYKData)
    Σ = Σ_SD(G, syk)
    prop_minus, prop_plus = propagtors(Σ, syk)
    return sqrt(det(prop_minus)), sqrt(det(prop_plus))
end


function G_SD(Σ, syk::SYKData)
    prop_minus, prop_plus = propagtors(Σ, syk)
    prop_minus_inv = -inv(prop_minus)'
    prop_ratio = prop_plus * prop_minus_inv
    det_ratio = det(prop_ratio)
    println("det_ratio = ", det_ratio)
	pfaff_ratio = sqrt(det_ratio)
	p_minus = 1 / (1 + pfaff_ratio)
    println("p_minus = ", p_minus)
	return p_minus * prop_minus_inv + (1 - p_minus) * -inv(prop_plus)'
end


function Σ_SD(G, syk::SYKData)
	return syk.J^2 * G.^(syk.q - 1)
end


function schwinger_dyson(G_init, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
    @assert iseven(syk.M)
    @assert iseven(syk.q)

	t = init_lerp

    G = G_init
	Σ = Σ_SD(G, syk)

    i = 1
    println("Iteration ", i)
    G_new = G_SD(Σ, syk)

    err = sum(abs.(G_new - G)) / sum(abs.(G))

	while i <= max_iters
		G_lerp = t * G_new + (1 - t) * G

		err_new = sum(abs.(G_lerp - G)) / sum(abs.(G))
		if isapprox(err_new, 0; atol=1e-10)
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

        p = plot(Gray.(G .- minimum(G)), title="Iteration $(i)")
        display(p)

        i += 1

        if i > max_iters
            println("Exceeded iterations!")
            break
        end

        println("Iteration ", i)
        G_new = G_SD(Σ, syk)
	end
    p = plot(Gray.(G .+ 0.5), title="β = $(syk.β)")
    display(p)
	return G, Σ
end


function action(G, Σ, syk::SYKData)
    L, M, Δτ = block_structure(G, syk)

    prop_minus, prop_plus = propagtors(Σ, syk)
    prop_term = -log(sqrt(det(prop_minus)) + sqrt(det(prop_plus)))
    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * Δτ^2 * sum(G.^syk.q)

    return syk.N * (prop_term + on_shell_term)
end


log_saddle(G, Σ, syk) = -action(G, Σ, syk)

log2_saddle(G, Σ, syk::SYKData) = log(2, ℯ) * log_saddle(G, Σ, syk)


function sre(G_M_init, G_2_init, G_Z_init, syk::SYKData; params_M = (0.5, 2, 1000), params_2 = (0.5, 2, 1000), params_Z = (0.5, 2, 1000))
    @assert iseven(syk.M)
    @assert iseven(syk.q)

    syk_M = syk
    syk_2 = SYKData(syk.N, syk.J, syk.q, 1, 2 * syk.β)
    syk_Z = SYKData(syk.N, syk.J, syk.q, 1, syk.β)

    _, L_M, _ = block_structure(G_M_init, syk_M)
    _, L_2, _ = block_structure(G_2_init, syk_2)
    _, L_Z, _ = block_structure(G_Z_init, syk_Z)

    @assert L_M == L_2 ÷ 2 == L_Z

    println("Computing ", syk.M, "-replica SRE saddle.")
    G_M, Σ_M = schwinger_dyson(G_M_init, syk_M; init_lerp = params_M[1], lerp_divisor = params_M[2], max_iters = params_M[3])
    log2_saddle_M = log2_saddle(G_M, Σ_M, syk_M)
    println("log2_saddle = ", log2_saddle_M)

    println("Computing purity saddle.")
    G_2, Σ_2 = SYKMatrix.schwinger_dyson(G_2_init, syk_2; init_lerp = params_2[1], lerp_divisor = params_2[2], max_iters = params_2[3])
    log2_saddle_2 = SYKMatrix.log2Z_saddle(G_2, Σ_2, syk_2) + 0.5
    println("log2_saddle = ", log2_saddle_2)

    println("Computing 1-replica SYK saddle.")
    G_Z, Σ_Z = SYKMatrix.schwinger_dyson(G_Z_init, syk_Z; init_lerp = params_Z[1], lerp_divisor = params_Z[2], max_iters = params_Z[3])
    log2_saddle_Z = SYKMatrix.log2Z_saddle(G_Z, Σ_Z, syk_Z)
    println("log2_saddle = ", log2_saddle_Z)

    sre = (log2_saddle_M - log2_saddle_2) / (1 - syk.M ÷ 2) + 2 * log2_saddle_Z
    println("sre = ", sre)
    
    return sre, G_M, G_2, G_Z
end

end
