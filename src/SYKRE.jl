module SYKRE

using CSV, DataFrames


include("MatrixSD.jl")


function matrix_sres(L, βs, α, q, N, J; max_iters=10000)
    steps = length(βs)
    sres = zeros(steps)
    # L, β, α. q, N, J
    Threads.@threads for i=1:steps
        println(i, " of ", steps)
        sres[i] = MatrixSD.sre_saddlepoint(L, βs[i], α, q, N, J, max_iters=max_iters)
        println("(T, SRE) = (", 1/βs[i], ", ", sres[i], ")")
    end

    return sres
end

q = 4
J = 1
α = 2
L = 100
T_min = 1
T_max = 2
steps = 10000
N = 1

Ts = collect(LinRange(T_min, T_max, steps))
βs = map(t -> 1/t, Ts)
sres = matrix_sres(L, βs, α, q, N, J)

df = DataFrame(T = Ts, β = βs, sre = sres)
CSV.write("./data/test.csv", df)


using Plots
data = CSV.read("./data/test.csv", DataFrame)
plot(data[!, 1], data[!, 3])



end
