using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.Replicas
using SYKRE.WeightReplicas
using SYKRE.WeightMatrix
using BenchmarkTools
using FFTW
using LinearAlgebra
using BlockArrays
using Random
using Combinatorics

# M = 4
# L = 1000

# R = Replicas.init(M, L)
# M_R = convert(Matrix{Float64}, R)

# @benchmark LinearAlgebra.inv(M_R)

# D = Replicas.differentials(4, 10)
# convert(Matrix{Float64}, D)

# Replicas.det(Replicas.inv(D))

# convert(Matrix{Float64}, Replicas.inv(D))

# LinearAlgebra.inv(M_R)

# @benchmark Replicas.inv(R)

# @benchmark Replicas.det(R)

# @benchmark LinearAlgebra.det(M_R)

# R_inv =  Replicas.inv(R)
# R_det = Replicas.det(R)
# M_inv = convert(Matrix{Float64}, R_inv)

# @benchmark [exp(im * π * (m - 1) / 4) for m = 1:4]
# @benchmark tmp = Matrix{ComplexF64}(undef, 1000, 1000)

# Replicas.inv(R)
# A = ones(ComplexF64, M, L, L)

# @benchmark real(A)

# test = ComplexF64.(ones(2, 2, 2))

# reinterpret(Float64, test)

# N = 1
# M = 2
# L = 1000
# β = 5
# J = 1
# q = 4

# w = 0.1

# syk = SYKData(N, J, q, M, β)

# G_init = Replicas.init(M, L)
# G_init_matrix = BlockedArray(convert(Matrix{Float64}, G_init), repeat([L], M), repeat([L], M))

# @time WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = 0.01, lerp_divisor = 2, max_iters=1000, tol=1e-5)

# @time SwapMatrix.schwinger_dyson(G_init_matrix, w, syk; init_lerp = 0.1, lerp_divisor = 2, max_iters=1000)


powervec(ss, n) = Iterators.product(ntuple(i->ss, n)...)

R = 10

T = zeros(Int, R, R)
T[1,R] = -1
for k = 2:R
    T[k, k-1] = 1
end
T

I_R = Matrix{Int}(I, R, R)
idx = collect(1:R)

perms = permutations(idx)

signs = powervec([1, -1], R)

for p in perms
    for s in signs
        P = I_R[p, :]
        S = diagm(collect(s))
        P = S * P
        T_perm = P * T * P'
        if T_perm * T == T * T_perm
            found = false
            for r = 1:R-1
                T_r = T^r
                if T_perm == T_r
                    println("T -> T^$(r)")
                    found = true
                    break
                elseif T_perm == -T_r
                    println("T -> -T^$(r)")
                    found = true
                    break
                # elseif T_perm == T_r'
                #     println("T -> (T^$(r))^T")
                #     found = true
                #     break
                # elseif T_perm == -T_r'
                #     println("T -> -(T^$(r))^T")
                #     found = true
                #     break
                end
            end
            if !found
                println("Not covered!")
                display(T_perm)
            end
        end
    end
end
