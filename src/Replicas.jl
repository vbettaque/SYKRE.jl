module Replicas

import Base: +, -, *, /, adjoint, transpose, convert, show, abs, inv, sum
using FFTW
using LinearAlgebra
using Plots

export ReplicaMatrix, block_diagonalize, det, frobenius

struct ReplicaMatrix
    R::Int
    L::Int
    blocks::Array{Float64, 3} # (L, L, R)
    bfft_plan::FFTW.Plan
end


function plot(A::ReplicaMatrix; title="")
    A_M = convert(Matrix{Float64}, A)
    blue = RGB(0,101.0/255,1)
    orange = RGB(1,154.0/255,0)
    grad = cgrad([blue, :gray95, orange], [0.0, 0.5, 1.0])
    p = heatmap(A_M, aspect_ratio = 1, clims=(-0.5, 0.5), yflip = true, color = grad, title=title)

    display(p)
end


function Base.:convert(::Type{Matrix{Float64}}, A::ReplicaMatrix)
    R = A.R
    L = A.L
    matrix = zeros(R*L, R*L)
    T = Matrix{Float64}(I, R, R)
    for i = 1:R
        matrix += kron(T, @view A.blocks[:, :, i])
        T = @view T[[R; 1:(R-1)], :]
        @view(T[1, (R - i + 1)]) .*= -1
    end
    return matrix
end


function Base.show(io::IO, obj::ReplicaMatrix)
    L_total = obj.R * obj.L
    println(io, L_total, "x", L_total, " ReplicaMatrix (R = ", obj.R, ", L = ", obj.L, "):")
    for i = 1:obj.R
        println(io, "Block (", i, ", 1): ", obj.blocks[:, :, i])
    end
end


function Base.show(io::IO, T::MIME"text/plain", obj::ReplicaMatrix)
    L_total = obj.R * obj.L
    println(io, L_total, "x", L_total, " ReplicaMatrix (R = ", obj.R, ", L = ", obj.L, "):")
    for i = 1:obj.R
        println(io, "Block (", i, ", 1): ")
        show(io, T, obj.blocks[:, :, i])
        println(io, "")
    end
end


function differentials(R, L; periodic = false)
    blocks = zeros(L, L, R)
    blocks[:, :, 1] .= Matrix{Float64}(I, L, L)
    Threads.@threads for i=1:L-1
        blocks[i+1, i, 1] = -1
    end
    blocks[1, L, 1] = periodic ? -1 : 1
    @views bfft_plan = plan_bfft!(ComplexF64.(blocks), 3; flags=FFTW.EXHAUSTIVE, timelimit=Inf)
    return ReplicaMatrix(R, L, blocks, bfft_plan)
end


function init(R, L)
    blocks = ones(L, L, R) ./ 2
    for j = 1:L
        for i = 1:j
            @view(blocks[i, j, 1]) .*= sign(i - j)
        end
    end
    bfft_plan = plan_bfft!(ComplexF64.(blocks), 3; flags=FFTW.EXHAUSTIVE, timelimit=Inf)
    return ReplicaMatrix(R, L, blocks, bfft_plan)
end


function Base.:(+)(A::ReplicaMatrix, B::ReplicaMatrix)
    @assert A.R == B.R && A.L == B.L
    return ReplicaMatrix(A.R, A.L, A.blocks .+ B.blocks, A.bfft_plan)
end


function Base.:(-)(A::ReplicaMatrix, B::ReplicaMatrix)
    @assert A.R == B.R && A.L == B.L
    return ReplicaMatrix(A.R, A.L, A.blocks .- B.blocks, A.bfft_plan)
end


function Base.:(-)(A::ReplicaMatrix)
    return ReplicaMatrix(A.R, A.L, - A.blocks, A.bfft_plan)
end


function Base.:(*)(A::ReplicaMatrix, B::ReplicaMatrix)
    R = A.R
    L = A.L
    @assert R == B.R && L == B.L
    new_blocks = zeros(L, L, R)
    for i=1:R
        for j=1:R
            idx_A = mod(i - j, R) + 1
            idx_B = j
            sgn = (i == j) ? 1 : sign(i - j)
            @views new_blocks[:, :, i] .+= sgn * A.blocks[:, :, idx_A] * B.blocks[:, :, idx_B]
        end
    end
    return ReplicaMatrix(R, L, new_blocks, A.bfft_plan)
end


function Base.:(*)(a::Real, B::ReplicaMatrix)
    return ReplicaMatrix(B.R, B.L, a * B.blocks, B.bfft_plan)
end


function Base.:(/)(A::ReplicaMatrix, b::Real)
    return ReplicaMatrix(A.R, A.L, A.blocks / b, A.bfft_plan)
end


function Base.broadcasted(::typeof(^), A::ReplicaMatrix, b::Real)
    @assert isodd(b)
    return ReplicaMatrix(A.R, A.L, A.blocks.^b, A.bfft_plan)
end


function Base.transpose(A::ReplicaMatrix)
    new_blocks = Array{Float64}(undef, A.L, A.L, A.R)
    @views new_blocks[:, :, 1] = transpose(A.blocks[:, :, 1])
    for i = 2:A.R
        i_conj = A.R + 2 - i
        @views new_blocks[:, :, i] = -transpose(A.blocks[:, :, i_conj])
    end
    return ReplicaMatrix(A.R, A.L, new_blocks, A.bfft_plan)
end


function Base.adjoint(A::ReplicaMatrix)
    return Base.transpose(A)
end

# Requires that f does not depend on the sign of the entry
function Base.sum(f, A::ReplicaMatrix)
    return A.R * sum(f, A.blocks)
end


function frobenius(A::ReplicaMatrix)
    return sqrt(A.R * sum(abs2, A.blocks))
end


function block_diagonalize(A::ReplicaMatrix)
    temp = Array{ComplexF64, 3}(undef, A.L, A.L, A.R)
    phases = [exp(im * π * (m - 1) / A.R) for m = 1:A.R]
    for i = 1:A.R
        @views mul!(temp[:, :, i], A.blocks[:, :, i], I, phases[i], 0)
    end
    return mul!(temp, A.bfft_plan, temp)
end


inv!(A) = LinearAlgebra.inv!(lu!(A))


function Base.inv(A::ReplicaMatrix)
    if isone(A.R)
        temp = Array{Float64}(undef, A.L, A.L, 1)
        @views temp[:, :, 1] = inv(A.blocks[:, :, 1])
        return ReplicaMatrix(A.R, A.L, temp, A.bfft_plan)
    end
    temp = ComplexF64.(A.blocks)
    phases = [exp(im * π * (m - 1) / A.R) for m = 1:A.R]
    for i = 1:A.R
        @views rmul!(temp[:, :, i], phases[i])
    end
    mul!(temp, A.bfft_plan, temp)
    for i = 1:floor(Int32, A.R / 2)
        @views temp[:, :, A.R + 1 - i] = conj(inv!(temp[:, :, i]))
    end
    @views isodd(A.R) && inv!(temp[:, :, (A.R + 1) ÷ 2])
    ldiv!(temp, A.bfft_plan, temp)
    for i = 1:A.R
        @views rdiv!(temp[:, :, i], phases[i])
    end
    return ReplicaMatrix(A.R, A.L, real(temp), A.bfft_plan)
end # 95.62 MiB, allocs estimate: 122


function det(A::ReplicaMatrix)
    if isone(A.R)
        return LinearAlgebra.det(@view(A.blocks[:, :, 1]))
    end
    diag = block_diagonalize(A)
    det = 1
    for i = 1:floor(Int32, A.R / 2)
        det *= abs2(LinearAlgebra.det(@view diag[:, :, i]))
    end
    if isodd(A.R)
        det *= real(LinearAlgebra.det(@view diag[:, :, (A.R + 1) ÷ 2]))
    end
    return det
end


end
