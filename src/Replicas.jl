module Replicas

import Base: +, -, *, /, adjoint, transpose, convert, show, abs, inv, sum
using FFTW
using LinearAlgebra
using Base.Threads

export ReplicaMatrix, block_diagonalize, det, frobenius

struct ReplicaMatrix
    M::Int
    L::Int
    blocks::Array{Float64, 3} # (L, L, M)
    bfft_plan::FFTW.Plan
end


function Base.:convert(::Type{Matrix{Float64}}, A::ReplicaMatrix)
    M = A.M
    L = A.L
    matrix = zeros(M*L, M*L)
    R = Matrix{Float64}(I, M, M)
    for i = 1:M
        matrix += kron(R, @view A.blocks[:, :, i])
        R = @view R[[M; 1:(M-1)], :]
        @view(R[1, (M - i + 1)]) .*= -1
    end
    return matrix
end


function Base.show(io::IO, obj::ReplicaMatrix)
    L_total = obj.M * obj.L
    println(io, L_total, "x", L_total, " ReplicaMatrix (M = ", obj.M, ", L = ", obj.L, "):")
    for i = 1:obj.M
        println(io, "Block (", i, ", 1): ", obj.blocks[:, :, i])
    end
end


function Base.show(io::IO, T::MIME"text/plain", obj::ReplicaMatrix)
    L_total = obj.M * obj.L
    println(io, L_total, "x", L_total, " ReplicaMatrix (M = ", obj.M, ", L = ", obj.L, "):")
    for i = 1:obj.M
        println(io, "Block (", i, ", 1): ")
        show(io, T, obj.blocks[:, :, i])
        println(io, "")
    end
end


function differentials(M, L; periodic = false)
    blocks = zeros(L, L, M)
    blocks[:, :, 1] .= Matrix{Float64}(I, L, L)
    Threads.@threads for i=1:L-1
        blocks[i+1, i, 1] = -1
    end
    blocks[1, L, 1] = periodic ? -1 : 1
    bfft_plan = plan_bfft!(ComplexF64.(blocks), 3; flags=FFTW.EXHAUSTIVE, timelimit=Inf)
    return ReplicaMatrix(M, L, blocks, bfft_plan)
end


function init(M, L)
    blocks = ones(L, L, M) / 2
    for j = 1:L
        for i = 1:j
            @view(blocks[i, j, 1]) .*= sign(i - j)
        end
    end
    bfft_plan = plan_bfft!(ComplexF64.(blocks), 3; flags=FFTW.EXHAUSTIVE, timelimit=Inf)
    return ReplicaMatrix(M, L, blocks, bfft_plan)
end


function init2(M, L)
    blocks = ones(L, L, M) / 2
    for j = 1:L
        for i = 1:j
            @view(blocks[i, j, 1]) .*= sign(i - j)
        end
    end
    bfft_plan = plan_bfft!(ComplexF64.(blocks), 3; flags=FFTW.EXHAUSTIVE, timelimit=Inf)
    return ReplicaMatrix(M, L, blocks, bfft_plan)
end


function Base.:(+)(A::ReplicaMatrix, B::ReplicaMatrix)
    @assert A.M == B.M && A.L == B.L
    return ReplicaMatrix(A.M, A.L, A.blocks .+ B.blocks, A.bfft_plan)
end


function Base.:(-)(A::ReplicaMatrix, B::ReplicaMatrix)
    @assert A.M == B.M && A.L == B.L
    return ReplicaMatrix(A.M, A.L, A.blocks .- B.blocks, A.bfft_plan)
end


function Base.:(-)(A::ReplicaMatrix)
    return ReplicaMatrix(A.M, A.L, - A.blocks, A.bfft_plan)
end


function Base.:(*)(A::ReplicaMatrix, B::ReplicaMatrix)
    M = A.M
    L = A.L
    @assert M == B.M && L == B.L
    new_blocks = zeros(L, L, M)
    for i=1:M
        for j=1:M
            idx_A = mod(i - j, M) + 1
            idx_B = j
            sgn = (i == j) ? 1 : sign(i - j)
            @views new_blocks[:, :, i] .+= sgn * A.blocks[:, :, idx_A] * B.blocks[:, :, idx_B]
        end
    end
    return ReplicaMatrix(M, L, new_blocks, A.bfft_plan)
end


function Base.:(*)(a::Real, B::ReplicaMatrix)
    return ReplicaMatrix(B.M, B.L, a * B.blocks, B.bfft_plan)
end


function Base.:(/)(A::ReplicaMatrix, b::Real)
    return ReplicaMatrix(A.M, A.L, A.blocks / b, A.bfft_plan)
end


function Base.broadcasted(::typeof(^), A::ReplicaMatrix, b::Real)
    @assert isodd(b)
    return ReplicaMatrix(A.M, A.L, A.blocks.^b, A.bfft_plan)
end


function Base.transpose(A::ReplicaMatrix)
    new_blocks = Array{Float64}(undef, A.L, A.L, A.M)
    @views new_blocks[:, :, 1] = transpose(A.blocks[:, :, 1])
    Threads.@threads for i = 2:A.M
        i_conj = A.M + 2 - i
        @views new_blocks[:, :, i] = -transpose(A.blocks[:, :, i_conj])
    end
    return ReplicaMatrix(A.M, A.L, new_blocks, A.bfft_plan)
end


function Base.adjoint(A::ReplicaMatrix)
    return Base.transpose(A)
end

# Requires that f does not depend on the sign of the entry
function Base.sum(f, A::ReplicaMatrix)
    return A.M * sum(f, A.blocks)
end


function frobenius(A::ReplicaMatrix)
    return sqrt(A.M * sum(abs2, A.blocks))
end


function block_diagonalize(A::ReplicaMatrix)
    temp = Array{ComplexF64, 3}(undef, A.L, A.L, A.M)
    phases = [exp(im * π * (m - 1) / A.M) for m = 1:A.M]
    Threads.@threads for i = 1:A.M
        @views mul!(temp[:, :, i], A.blocks[:, :, i], I, phases[i], 0)
    end
    return mul!(temp, A.bfft_plan, temp)
end


inv!(A) = LinearAlgebra.inv!(lu!(A))


function Base.inv(A::ReplicaMatrix)
    temp = Array{ComplexF64, 3}(undef, A.L, A.L, A.M)
    phases = [exp(im * π * (m - 1) / A.M) for m = 1:A.M]
    Threads.@threads for i = 1:A.M
        @views mul!(temp[:, :, i], A.blocks[:, :, i], I, phases[i], 0)
    end
    mul!(temp, A.bfft_plan, temp)
    for i = 1:A.M
        inv!(@view temp[:, :, i])
    end
    ldiv!(temp, A.bfft_plan, temp)
    Threads.@threads for i = 1:A.M
        rdiv!(@view(temp[:, :, i]), phases[i])
    end
    return ReplicaMatrix(A.M, A.L, real.(temp), A.bfft_plan)
end # 95.62 MiB, allocs estimate: 122


function det(A::ReplicaMatrix)
    diag = block_diagonalize(A)
    det = 1
    for i = 1:A.M
        det *= LinearAlgebra.det(@view diag[:, :, i])
    end
    return real(det)
end


end
