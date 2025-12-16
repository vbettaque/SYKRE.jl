using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays, Combinatorics, BenchmarkTools

using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas

function binary_entropy(w)
    @assert 0 ≤ w ≤ 1
    if iszero(w) || isone(w)
        return 0
    end
    return - w * log2(w) - (1 - w) * log2(1 - w)
end

function generate_2R1_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    G_init = Replicas.init(1, L)

    syk = SYKData(1, 1, q, 1, β)

    G_temp = Replicas.init(1, L)
    Σ_temp = Replicas.init(1, L)

    path = "data/weighted/2R1/"
    filename = "weighted_2R1" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "weight,saddle,pf_minus,pf_plus,p_plus,entropy\n")
    end

    for i in eachindex(ws)
        w = ws[i]
        println(i, " out of ", length(ws), ": w = ", w)

        saddle = 0
        pf_minus = 0
        pf_plus = 0
        p_plus = 0
        entropy = 0

        G_temp, Σ_temp = WeightedReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        saddle = 2 * WeightedReplicas.log2_saddle(G_temp, Σ_temp, 0, syk)
        pf_minus, pf_plus = WeightedReplicas.pfaffians(Σ_temp, syk)
        pf_minus *= pf_minus
        pf_plus *= pf_plus
        p_plus = pf_plus / (pf_plus + pf_minus)
        entropy = binary_entropy(w)



        df = DataFrame(weight = w, saddle = saddle, pf_minus = pf_minus, pf_plus = pf_plus, p_plus = p_plus, entropy = entropy)

        CSV.write(file, df, append=true)
    end

end

function generate_1R2_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    G_init = Replicas.init(2, L)

    syk = SYKData(1, 1, q, 2, β)

    G_temp = Replicas.init(2, L)
    Σ_temp = Replicas.init(2, L)

    path = "data/weighted/1R2/"
    filename = "weight_1R2" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "weight,saddle,pf_minus,pf_plus,p_plus,entropy\n")
    end

    for i in eachindex(ws)
        w = ws[i]
        @info "$(i) out of $(length(ws)): w = $(w)"

        saddle = 0
        pf_minus = 0
        pf_plus = 0
        p_plus = 0
        entropy = 0

        G_temp, Σ_temp = WeightedReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        saddle = WeightedReplicas.log2_saddle(G_temp, Σ_temp, w, syk)
        pf_minus, pf_plus = WeightedReplicas.pfaffians(Σ_temp, syk)
        p_plus = pf_plus / (pf_plus + pf_minus)
        entropy = binary_entropy(w)

        df = DataFrame(weight = w, saddle = saddle, pf_minus = pf_minus, pf_plus = pf_plus, p_plus = p_plus, entropy = entropy)

        # Replicas.plot(G_temp, title = "β = $(β), w = $(w)")

        CSV.write(file, df, append=true)
    end

end

L = 1000
ws = collect(0:2:100) / 100


generate_1R2_weight_data([0.5], L, 20.0, 4; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)

# for β in [10.0, 20.0, 50.0]
#     for q in [2, 4, 6, 8]
#         β == 10.0 && q == 2 && continue
#         generate_1R2_weight_data(ws, L, β, q; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
#     end
# end
