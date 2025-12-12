using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays, Combinatorics

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.Replicas
using SYKRE.WeightReplicas
using SYKRE.PurityReplicas
using SYKRE.WeightMatrix


# G_1R4_init = Replicas.init(4, L)
# @views G_1R4_init.blocks[:, :, 3] .= 0

# G_1R6a_init = Replicas.init(6, L)
# @views G_1R6a_init.blocks[:, :, 3] .= 0
# @views G_1R6a_init.blocks[:, :, 5] .= 0

# G_1R6b_init = Replicas.init(6, L)
# @views G_1R6b_init.blocks[:, :, 4] .= -1

# G_1R8a_init = Replicas.init(8, L)
# @views G_1R8a_init.blocks[:, :, 3] .= 0
# @views G_1R8a_init.blocks[:, :, 5] .= 0
# @views G_1R8a_init.blocks[:, :, 7] .= 0

# G_1R8b_init = Replicas.init(8, L)
# @views G_1R8b_init.blocks[:, :, 5] .= -1


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

    path = "data/paper/weights/2R1/"
    filename = "weight_2R1" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
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

        G_temp, Σ_temp = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        saddle = 2 * WeightReplicas.log2_saddle(G_temp, Σ_temp, 0, syk)
        pf_minus, pf_plus = WeightReplicas.pfaffians(Σ_temp, syk)
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

    path = "data/paper/weights/1R2/"
    filename = "weight_1R2" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
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

        G_temp, Σ_temp = WeightReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        saddle = WeightReplicas.log2_saddle(G_temp, Σ_temp, w, syk)
        pf_minus, pf_plus = WeightReplicas.pfaffians(Σ_temp, syk)
        p_plus = pf_plus / (pf_plus + pf_minus)
        entropy = binary_entropy(w)

        df = DataFrame(weight = w, saddle = saddle, pf_minus = pf_minus, pf_plus = pf_plus, p_plus = p_plus, entropy = entropy)

        CSV.write(file, df, append=true)
    end

end

L = 1000
ws = collect(0:2:100) / 100

generate_1R2_weight_data(collect(68:2:100) / 100, L, 10.0, 2; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)


for β in [10.0, 20.0, 50.0]
    for q in [2, 4, 6, 8]
        β == 10.0 && q == 2 && continue
        generate_1R2_weight_data(ws, L, β, q; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    end
end
