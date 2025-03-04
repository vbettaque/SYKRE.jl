module SYKRE

using CSV, DataFrames


include("MatrixSD.jl")


function matrix_sre(q, J, α, N, L, β_min, β_max, steps; max_iters=10000)
    βs = LinRange(β_min, β_max, steps)
    sres = zeros(steps)

    for i=1:steps
        println(i, " of ", steps)
        Σ, G = MatrixSD.sre_schwinger_dyson(q, J, βs[i], α, L; max_iters)
        # display(Σ)
        # display(G)
        sres[i] = MatrixSD.sre_saddlepoint(Σ, G, βs[i], N, q, J, α)
        println(sres[i])
    end

    return βs, sres
end

q = 4
J = 1
α = 2
L = 1000
β_min = 0.01
β_max = 1
steps = 100
N = 1000


βs, sres = matrix_sre(q, J, α, N, L, β_min, β_max, steps)


df = DataFrame(β = βs, sre = sres)
CSV.write("./data/test.csv", df)



end