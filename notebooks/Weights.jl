using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.Replicas
using SYKRE.WeightReplicas


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
    G_init = Replicas.init(syk.M, L)

    t = min(inv(2 * syk.β * syk.J), 0.01)
    tol = (syk.β * syk.J / L)^2

    G, Σ = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = t, lerp_divisor = 2, tol=tol, max_iters=10000)

    pf_minus, pf_plus = WeightReplicas.pfaffians(Σ, syk)
    p_plus = pf_plus / (pf_plus + pf_minus)
    return p_plus - w
end

###################################3

N = 1
J = 1
q = 4
M = 4
L = 500

βs = LinRange(0.4, 10, 97)
ws = zeros(length(βs))

path = "data/sre_weights/"
filename = "weights_M" * string(M) * "_q" * string(q) * "_L" * string(L) * ".csv"
file = path * filename
!ispath(path) && mkdir(path)
if !isfile(file)
    touch(file)
    write(file, "β,weight,weight_lower,weight_upper\n")
end

w_min = 0.
w_max = 0.01

for i in eachindex(βs)
    β = βs[i]
    println(i, " out of ", length(ws), ": β = ", β)

    syk = SYKData(N, J, q, M, β)
    f(w) = weight_difference(w, L, syk)

    w, w_lower, w_upper = bisection(f, w_min, w_max; tol=0.00001, max_iter=100)

    df = DataFrame(β = β, weight = w, weight_lower = w_lower, weight_upper = w_upper)
    CSV.write(file, df, append=true)

    global w_min = w_lower
end

weights_data = CSV.File("data/sre_weights/weights_M2_q4_L1000.csv") |> DataFrame

# p = plot(weights_data[:,1], weights_data[:,2], label="q=4, M=2, L=1000")
# xlabel!("β")
# ylabel!("w_crit")
# display(p)

###########################################3

# N = 1
# J = 1
# q = 4
# M = 4
# β = 5
# Ls = collect(1:50) .* 10
# weights = zeros(size(Ls))

# syk = SYKData(N, J, q, M, β)

# for i in eachindex(Ls)
#     L = Ls[i]
#     println(i, " out of ", length(Ls), ": L = ", L)
#     f(w) = weight_difference(w, L, syk)
#     w, w_lower, w_upper = bisection(f, 0, 0.5; tol=0.0001, max_iter=100)
#     weights[i] = w
# end

# p = plot(Ls, weights, label = "M = 4, q = 4, β = 5")
# xaxis!("L")
# yaxis!("weight")

# display(p)
