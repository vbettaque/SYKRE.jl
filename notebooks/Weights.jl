using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.SwapMatrix

function bisection(f, a, b; tol=0.0001, max_iter=100)
    @assert a < b
    fa = f(a)
    fb = f(b)
    @assert fa * fb < 0

    i = 1
    while i ≤ max_iter
        println("a = ", a, ", b = ", b)
        println("f(a) = ", fa, ", f(b) = ", fb)
        c = (a + b) / 2
        println("c = ", c)
        if (b - a) / 2 < tol
            return c, a, b
        end
        fc = f(c)
        println("f(c) = ", fc)
        if iszero(fc)
            return c, a, b
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
    G_init = inv(SwapMatrix.differential(syk.M * L)) - 0.5 * I
    G_init = BlockedArray(G_init, repeat([L], syk.M), repeat([L], syk.M))

    G, Σ = SwapMatrix.schwinger_dyson(G_init, w, syk; init_lerp = 0.01, lerp_divisor = 2, max_iters=10000)

    pf_minus, pf_plus = SwapMatrix.pfaffians(Σ, syk)
    p_plus = pf_plus / (pf_plus + pf_minus)
    return p_plus - w
end

N = 1
J = 1
q = 4
M = 4
L = 250

βs = Float64.(10:10)
ws = zeros(length(βs))

path = "data/sre_weights/"
filename = "weights_M" * string(M) * "_q" * string(q) * "_L" * string(L) * ".csv"
file = path * filename
!ispath(path) && mkdir(path)
if !isfile(file)
    touch(file)
    write(file, "β,weight,weight_lower,weight_upper\n")
end

for i in eachindex(βs)
    β = βs[i]
    println(i, " out of ", length(ws), ": β = ", β)

    syk = SYKData(N, J, q, M, β)
    f(w) = weight_difference(w, L, syk)

    w, w_lower, w_upper = bisection(f, 0.0, 0.5)

    df = DataFrame(β = β, weight = w, weight_lower = w_lower, weight_upper = w_upper)
    CSV.write(file, df, append=true)
end
