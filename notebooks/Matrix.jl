using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier

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

function generate_matrix_sre_data(L, α, q, βs)
    N = 1
    J = 1

    path = "data/sre_matrix/"
    filename = "sre" * string(α) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "β,sre,pf_minus_α,pf_plus_α,pf_minus_1,pf_plus_1\n")
    end

    Σ_init = J * inv(SREMatrix.differential(L)).^(q-1)

    for i in eachindex(βs)
        β = βs[i]
        println(i, ": β = ", β)
        syk = SYK.SYKData(N, J, q, 2*α, β)

        sre = 0
        pf_minus_α = 0
        pf_plus_α = 0
        pf_minus_1 = 0
        pf_plus_1 = 0

        try
            sre, Σ_M, Σ_2, Σ_Z =
                SREMatrix.sre(L, syk; Σ_M_init=Σ_init, Σ_2_init=Σ_init, Σ_Z_init=Σ_init,max_iters=500)

            pf_minus_α, pf_plus_α = SREMatrix.pfaffians(Σ_M, syk)
            pf_minus_1, pf_plus_1 = SREMatrix.pfaffians(Σ_2, SYK.SYKData(N, J, q, 2, β))

            display(Σ_2)

        catch e
            display(e)
            sre = NaN
            pf_minus_α = NaN
            pf_plus_α = NaN
            pf_minus_1 = NaN
            pf_plus_1 = NaN
        end

        df = DataFrame(β = β, sre = sre, pf_minus_α = pf_minus_α, pf_plus_α = pf_plus_α, pf_minus_1 = pf_minus_1, pf_plus_1 = pf_plus_1)
        CSV.write(file, df, append=true)
    end

end

N = 1
J = 1.
q = 4
M = 4
L = 4000
L_rep = L ÷ M
β = 50

syk = SYKData(N, J, q, M, β)

R_minus = Matrix{Float64}(I, M, M)
R_plus = [0 -1 -1 -1; 1 0 -1 -1; 1 1 0 -1; 1 1 1 0]

Σ_minus = zeros(L_rep, L_rep)
rands = rand(Float64, L_rep) / 2 
rands[2:L_rep] = (rands[2:L_rep] + reverse(rands[2:L_rep]))/2

eachindex(Σ_minus)
for i = 1:L_rep
    for j = 1:L_rep
        Σ_minus[i, j] = sign(i - j) * rands[abs(i - j) + 1]
        i == j && (Σ_minus[i, i] = 0.5)
    end
end

Σ_plus = abs.(Σ_minus)
Σ_init = kron(R_minus, Σ_minus) + kron(R_plus, Σ_plus)

plot(Gray.(Σ_init .- minimum(Σ_init)))

# Σ_init = rand(Float64, (L, L))
# Σ_init = 0.5 * I + (Σ_init - Σ_init')/4

# logZ, Σ, G = SYKMatrix.logZ(L, SYKData(N, J, q, 1, β); Σ_init=Σ_init, max_iters=1000)
sre, Σ, G = SREMatrix.logZ(L, syk; Σ_init=Σ_init, max_iters=1000)

τs = []
G_invariant = zeros(2 * L_rep, M, M)

p = plot()

for i=1:M
    for j=1:M
        i_range = ((i-1) * L_rep + 1):(i * L_rep)
        j_range = ((j-1) * L_rep + 1):(j * L_rep)
        τs, G_invariant[:, i, j] = time_invariance(G[i_range, j_range], β)
        p = plot!(τs, G_invariant[:, i, j], label="G_$(i)$(j)")
    end
end

display(p)

plot()
plot!(τs, G_invariant[:, 1, 1], label="G_11")
plot!(τs, G_invariant[:, 2, 2], label="G_22")
plot!(τs, G_invariant[:, 3, 3], label="G_33")
plot!(τs, G_invariant[:, 4, 4], label="G_44")

plot!(τs, G_invariant[:, 1, 2], label="G_12")
plot!(τs, G_invariant[:, 2, 3], label="G_23")
plot!(τs, G_invariant[:, 3, 4], label="G_34")
plot!(τs, G_invariant[:, 1, 4], label="G_14")
plot!(τs, -G_invariant[:, 2, 1], label="-G_21")
plot!(τs, -G_invariant[:, 3, 2], label="-G_32")
plot!(τs, -G_invariant[:, 4, 3], label="-G_43")
plot!(τs, -G_invariant[:, 4, 1], label="-G_41")

plot!(τs, G_invariant[:, 1, 3], label="G_13")
plot!(τs, G_invariant[:, 2, 4], label="G_24")
plot!(τs, -G_invariant[:, 3, 1], label="-G_13")
plot!(τs, -G_invariant[:, 4, 2], label="-G_42")



plot(Gray.(G .+ 0.5))

Σ

sre

sqrt(det(SREMatrix.differential(L) - (β/L)^2 * Σ))

sqrt(det(kron(R_minus, SREMatrix.differential(L_rep)) - (β/L_rep)^2 * Σ))


function inverse_determinant_replicas(R, M)
	@assert M > 0 && isinteger(log2(M))
	L, L_ = size(R)
	@assert L == L_ && iszero(L % M)
	L_2 = L ÷ 2

	A = R[1:L_2, 1:L_2]
	B = R[L_2+1:L, 1:L_2]
	@assert A ≈ R[L_2+1:L, L_2+1:L]
	@assert B ≈ -R[1:L_2, L_2+1:L]
	
	A_inv, det_A = (M == 2 ? (inv(A), det(A)) : inverse_determinant_replicas(A, M÷2))
	schur = A + B * A_inv * B
	schur_inv = inv(A + B * A_inv * B)
	det_schur = det(schur)
	off_diag = schur_inv * B * A_inv
	
	return ([schur_inv off_diag; -off_diag schur_inv], det_A * det_schur)
end