using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas
using BenchmarkTools
using FFTW
using LinearAlgebra
using BlockArrays
using Random
using Combinatorics


powervec(ss, n) = Iterators.product(ntuple(i->ss, n)...)

R = 6

# T = zeros(Int, R, R)
# T[1,R] = -1
# for k = 2:R
#     T[k, k-1] = 1
# end
# T

# I_R = Matrix{Int}(I, R, R)
# idx = collect(1:R)

# perms = permutations(idx)

# signs = powervec([1, -1], R)

# for p in perms
#     for s in signs
#         P = I_R[p, :]
#         S = diagm(collect(s))
#         P = S * P
#         T_perm = P * T * P'
#         if T_perm * T == T * T_perm
#             found = false
#             for r = 1:R-1
#                 T_r = T^r
#                 if T_perm == T_r
#                     println("T -> T^$(r)")
#                     found = true
#                     break
#                 elseif T_perm == -T_r
#                     println("T -> -T^$(r)")
#                     found = true
#                     break
#                 # elseif T_perm == T_r'
#                 #     println("T -> (T^$(r))^T")
#                 #     found = true
#                 #     break
#                 # elseif T_perm == -T_r'
#                 #     println("T -> -(T^$(r))^T")
#                 #     found = true
#                 #     break
#                 end
#             end
#             if !found
#                 println("Not covered!")
#                 display(T_perm)
#             end
#         end
#     end
# end


function flip(v)
    v_flipped = zeros(Int, length(v))
    for i in eachindex(v)
        v_flipped[i] = (-1)^(i-1) * v[i]
    end
    return v_flipped
end

function permute_reps(v, k)
    R = length(v)
    R_2 = R ÷ 2
    @assert 1 ≤ k ≤ R_2
    isodd(R_2) && @assert 2*k - 1 != R_2
    v_perm = zeros(Int, R)
    for r in eachindex(v)
        sign = (-1)^floor(Int, (2*k - 1) * (r-1) / R)
        index = ((2*k - 1) * (r-1) % R) + 1
        v_perm[r] = sign * v[index]
    end
    v_perm
end


function equal_reps(v, w)
    v = collect(v)
    w = collect(w)
    R = length(v)
    @assert length(w) == R
    R_2 = R ÷ 2

    v == w && return true
    v == -w && return true
    v == flip(w) && return true
    v == -flip(w) && return true

    for k = 2:R_2
        isodd(R_2) && (2 * k - 1 == R_2) && continue
        w_perm = permute_reps(w, k)
        v == w_perm && return true
        v == -w_perm && return true
        v == flip(w_perm) && return true
        v == -flip(w_perm) && return true
    end

    return false
end


function get_unique_reps(R)
    reps = powervec([0, 1, -1], R)
    unique_reps = []

    for v in reps
        iszero(v[1]) && continue
        v[2:end] != reverse(v[2:end]) && continue

        is_unique = true
        for w in unique_reps
            if equal_reps(v, w)
                is_unique = false
                break
            end
        end
        is_unique && push!(unique_reps, v)
    end

    return unique_reps[2:end]
end

N = 1
J = 1
q = 4
β = 20
L = 1000
w = 0.5

R = 8

syk = SYKData(N, J, q, R, β)

for rep in get_unique_reps(R)
    G_init = Replicas.init(R, L)
    for i = 1:R
        @views G_init.blocks[:, :, i] *= rep[i]
    end
    Replicas.plot(G_init; title="$(rep)")
    G, Σ = WeightedReplicas.schwinger_dyson(G_init, w, syk; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    log_saddle = WeightedReplicas.log_saddle(G, Σ, w, syk)
    Replicas.plot(G; title="log_saddle = $(log_saddle)")
end
