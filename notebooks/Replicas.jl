using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.Replicas

M = 4
L = 1000

R1 = ReplicaMatrix(M, L, [rand(Float64, L, L) for _ in 1:M])
M1 = convert(Matrix{Float64}, R1)

R2 = ReplicaMatrix(M, L, [rand(Float64, L, L) for _ in 1:M])
M2 = convert(Matrix{Float64}, R2)

@time Mres = M1 * M2
@time Rres = R1 * R2

Mres - Rres

@assert(Mres ≈ Rres)


convert(Matrix{Float64}, R1)