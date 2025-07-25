module SREMatrixReplica

using BlockArrays
using LinearAlgebra
using Plots

using ..SYK
using ..SYKMatrix

function differential(L; M = 1, periodic=false)
    D = Matrix{Float64}(I, L, L)
    for i=1:L-1
        D[i+1, i] = -1
    end
    D[1, L] = periodic ? -1 : 1
    I_M = Matrix{Float64}(I, M, M)
	return BlockKron(I_M, D)
end

function inverse_and_pfaffian(A)
    Ls = blocksizes(Σ)
    @assert allequal(Ls)
    L, L_ = Ls[1, 1]
    @assert L == L_
    M, M_ = size(Ls)
    @assert isinteger(log2(M)) && M == M_
    M_2 = M ÷ 2

    A_diag = view(A, Block(1:M_2, 1:M_2))
    @assert blockisequal(A_diag, view(A, Block(1:M_2, 1:M_2)))
    A_off = view(A, Block(M_2+1:M, 1:M÷2))
    @assert blockisequal(A_off, -view(A, Block(1:M_2, M_2+1:M)))

	A_inv, pf_A = (M == 2 ? (inv(A), sqrt(det(A))) : inverse_pfaffian(A_diag))
	
    schur = A_diag + A_off * A_inv * A_off
	schur_inv, pf_schur = (M == 2 ? (inv(schur), sqrt(det(schur))) : inverse_pfaffian(schur))

	inv_off_diag = schur_inv * B * A_inv
	
	return mortar([[schur_inv inv_off_diag; -inv_off_diag schur_inv]]), pf_A * pf_schur
end



function G_SD(Σ, syk::SYKData)
    Ls = blocksizes(Σ)
    @assert allequal(Ls)
    L, L_ = Ls[1, 1]
    @assert L == L_
    M, M_ = size(Ls)
    @assert iseven(M) && M == M_ == syk.M

    Δτ = syk.β/L

	D_minus = differential(L; M = M, periodic=false)
    D_plus = differential(L; M = M, periodic=true)

    prop_minus = D_minus - Δτ^2 * Σ
	prop_plus = D_plus - Δτ^2 * Σ

    inv_minus, pf_minus = undef
    inv

    prop_ratio = prop_minus * inv(prop_plus)
    det_ratio = det(prop_ratio)
    det_ratio < 0 && return NaN
	pfaff_ratio = sqrt(det(prop_ratio))
    println("pfaff_ratio = ", pfaff_ratio)
	p_plus = 1 / (1 + pfaff_ratio)
    println(p_plus)
	return (1 - p_plus) * inv(prop_minus) + p_plus * inv(prop_plus)
end


function Σ_SD(G, syk::SYKData)
	return syk.J^2 * G.^(syk.q - 1)
end

function time_invariance(M, β)
    L, _ = size(M)
    Δτ = β / L
    τs = LinRange(0, 2β - Δτ, 2 * L)
    M_invariant = zeros(2 * L)
    M_invariant[1:L] = M[1:L, 1]
    M_invariant[L+2:2L] = reverse(M[1, 2:L])
    M_invariant[L+1] = (M_invariant[L] + M_invariant[L+2])/2
    return τs, M_invariant
end

function schwinger_dyson(L, syk::SYKData; Σ_init = zeros(L, L), max_iters=1000)
    @assert iseven(syk.M)
    @assert iseven(syk.q)
    @assert iszero(L % syk.M)

	t = 0.1; b = 2; err=0
	Σ = Σ_init
	G = G_SD(Σ, syk)
    p = plot(Gray.(G .- minimum(G)), title="Iteration 0")
    display(p)
    Σ = Σ_SD(G, syk)
    i = 1
	while i <= max_iters
		G_new = t * G_SD(Σ, syk) + (1 - t) * G
        p = plot(Gray.(G_new .- minimum(G_new)), title="Iteration $(i)")
        display(p)
		err_new = sum(abs.(G_new - G)) / sum(abs.(G))
		if isapprox(err_new, 0; atol=1e-10)
            G = G_new
            Σ = Σ_SD(G, syk)
            println("Converged after ", i, " iterations")
            break
        end
		if (err_new > err && i > 1)
            t /= b
            i -= 1
            continue
        end
        Σ_new = Σ_SD(G_new, syk)
        if any(isnan.(G_SD(Σ_new, syk)))
            t /= b
            i -= 1
            continue
        end
		err = err_new
        println("err = ", err, " t = ", t)
		G = G_new
        
        Σ = Σ_new
        i == max_iters && println("Exceeded iterations!")
        i += 1
	end
	return Σ, G
end


function action(Σ, G, syk::SYKData)
    L, _ = size(Σ)
    @assert iseven(L)
    @assert iseven(syk.M)
    @assert iseven(syk.q)
    @assert iszero(L % syk.M)
    L_rep = L ÷ syk.M
    @assert iseven(L_rep)

    Δτ = syk.β/L_rep
    I_rep = Matrix{Float64}(I, syk.M, syk.M)
	D_minus = kron(I_rep, differential(L_rep))
    D_plus = kron(I_rep, differential(L_rep, anti_periodic=false))
    prop_minus = D_minus - Δτ^2 * Σ
	prop_plus = D_plus - Δτ^2 * Σ

    prop_term = -log(sqrt(det(prop_minus)) + sqrt(det(prop_plus)))
    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * Δτ^2 * sum(G.^syk.q)
    return syk.N * (prop_term + on_shell_term)
end

function logZ(L, syk::SYKData; Σ_init = zeros(L,L), max_iters=1000)
    Σ, G = schwinger_dyson(L, syk; Σ_init = Σ_init, max_iters=max_iters)
    return -action(Σ, G, syk), Σ, G
end

function sre(L, syk::SYKData; Σ_M_init = zeros(L, L), Σ_2_init = zeros(L, L), Σ_Z_init = zeros(L, L), max_iters=1000)
    @assert iseven(syk.M)
    @assert iseven(syk.q)
    @assert iseven(L)
    @assert iszero(L % syk.M)
    @assert iseven(L ÷ syk.M)
    @assert iseven(L ÷ 2)

    syk_M = syk
    syk_2 = SYKData(syk.N, syk.J, syk.q, 2, syk.β)
    syk_Z = SYKData(syk.N, syk.J, syk.q, 1, syk.β)

    Σ_2, G_2 = schwinger_dyson(L, syk_2; Σ_init=Σ_2_init, max_iters=max_iters)
    Σ_M, G_M = schwinger_dyson(L, syk_M; Σ_init=Σ_M_init, max_iters=max_iters)

    logtr_swap_M = - log(2, ℯ) * action(Σ_M, G_M, syk_M)
    logtr_swap_2 = - log(2, ℯ) * action(Σ_2, G_2, syk_2)
    logZ, Σ_Z = SYKMatrix.logZ(L, syk_Z; Σ_init=Σ_Z_init, max_iters=max_iters)

    println("logtr_swap_M = ", logtr_swap_M)
    println("logtr_swap_2 = ", logtr_swap_2)
    println("logZ = ", logZ)
    sre = (logtr_swap_M - logtr_swap_2) / (1 - syk.M/2) + 2 * log(2, ℯ) * logZ
    println("sre = ", sre)
    return sre, Σ_M, Σ_2, Σ_Z
end

end
