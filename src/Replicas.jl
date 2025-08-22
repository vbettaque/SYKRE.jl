module Replicas

import Base: *, display, convert
using FFTW
using LinearAlgebra

export ReplicaMatrix
    
struct ReplicaMatrix{T}
    M::Int
    L::Int
    blocks::Array{T, 3} # (L, L, M)
end

function init{T}(M, L)
    blocks = Array{T, 3}(undef, L, L, M)
    
end

function Base.:(*)(A::ReplicaMatrix{T}, B::ReplicaMatrix{T})
    M = A.M
    L = A.L
    @assert M == B.M && L == B.L
    new_blocks = zeros(T, (L, L, M))
    for i=1:M
        for j=1:M
            idx_A = mod(i - j, M) + 1
            idx_B = j
            sgn = (i == j) ? 1 : sign(i - j)
            new_blocks[:, :, i] += sgn * A.blocks[:, :, idx_A] * B.blocks[:, :, idx_B]
        end
    end
    return ReplicaMatrix(M, L, newblocks)
end

function Base.:convert(::Type{Matrix{T}}, A::ReplicaMatrix{T})
    M = A.M
    L = A.L
    matrix = zeros(T, (M*L, M*L))
    R = Matrix{Float64}(I, M, M)
    for i = 1:M
        matrix += kron(R, A.blocks[:, :, i])
        R = R[[M; 1:(M-1)], :]
        R[1, (M - i + 1)] *= -1
    end
    return matrix
end

function block_diagonalize(A::ReplicaMatrix{T})
    input = Array{T, 3}(undef, A.L, A.L, A.M)
    phases = [exp(im * π * (m - 1) / A.M) for m = 1:A.M]
    for i = 1:A.M
        input[:, :, i] = phases[i] * input[:, :, i]
    end
    return ReplicaMatrix{T}(A.M, A.L, bfft!(input, 3))
end

function block_undiagonalize(A::ReplicaMatrix{T})
    @assert L == L_
    inverse = fft(A.blocks, 3) / A.M
    phases = [exp(-im * π * (m - 1) / A.M) for m = 1:A.M]
    for i = 1:A.M
        inverse[:, :, i] = phases[i] * inverse[:, :, i]
    end
    @assert isreal(inverse)
    return ReplicaMatrix(A.M, A.L, real(inverse))
end

function LinearAlgebra.inv(A::ReplicaMatrix{T})

end

end