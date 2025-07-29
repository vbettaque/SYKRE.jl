module Replicas

import Base: *, display, convert
using StaticArrays
using LinearAlgebra

export ReplicaMatrix
    
struct ReplicaMatrix
    M::Int
    L::Int
    entries::Vector{Matrix{Float64}}
end

function Base.:(*)(A::ReplicaMatrix, B::ReplicaMatrix)
    M = A.M
    L = A.L
    @assert M == B.M && L == B.L
    new_entries = [zeros(L, L) for _ in 1:M]
    for i=1:M
        for j=1:M
            idx_A = mod(i - j, M) + 1
            idx_B = j
            sgn = (i == j) ? 1 : sign(i - j)
            new_entries[i] += sgn * A.entries[idx_A] * B.entries[idx_B]
        end
    end
    return ReplicaMatrix(M, L, new_entries)
end

function Base.:convert(::Type{Matrix{Float64}}, A::ReplicaMatrix)
    M = A.M
    L = A.L
    matrix = zeros(M*L, M*L)
    R = Matrix{Float64}(I, M, M)
    for i = 1:M
        matrix += kron(R, A.entries[i])
        R = R[[M; 1:(M-1)], :]
        R[1, (M - i + 1)] *= -1
    end
    return matrix
end

end