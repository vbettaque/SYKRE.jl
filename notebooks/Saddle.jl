using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.SwapMatrix

function bisection(f, a, b; tol=0.001, max_iter=100)
    @assert a < b
    fa = f(a)
    fb = f(b)
    @assert fa * fb < 0

    i = 1
    while i ≤ max_iter
        c = (a + b) / 2
        if (b - a) / 2 < tol
            return c
        end
        fc = f(c)
        if iszero(fc)
            return c
        end
        i += 1
        if sign(fc) == sign(fa)
            a = c
            fa = fc
        else
            b = c
            fb = fc
        end
    end
    println("Exceeded maximum number of steps!")
end

function weight_difference(w, L, syk::SYKData)
    G_init = inv(SYKMatrix.differential(syk.M * L))
    G_init = BlockedArray(G_init, repeat([L], syk.M), repeat([L], syk.M))
    G, Σ = SwapMatrix.schwinger_dyson(G_init, w, syk; init_lerp = 0.1, lerp_divisor = 10, max_iters=1000)
    pf_minus, pf_plus = SwapMatrix.pfaffians(Σ, syk)
    p_plus = pf_plus / (pf_plus + pf_minus)
    return p_plus - w
end

N = 1
J = 1
q = 4
M = 4
β = 10
L = 1000

syk = SYKData(N, J, q, M, β)

f(w) = weight_difference(w, L, syk)

bisection(f, 0, 0.1)