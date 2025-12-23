using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays, Combinatorics, BenchmarkTools, Profile

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
    !ispath(path) && mkpath(path)
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
    !ispath(path) && mkpath(path)
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

        Replicas.plot(G_temp, title = "β = $(β), w = $(w)")

        CSV.write(file, df, append=true)
    end

end


function generate_1R4a_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    # G_init_single = Replicas.init(1, 4*L)
    # G_init_single, _ = WeightReplicas.schwinger_dyson(G_init_single, 0.0, SYKData(1, 1, q, 1, 4*β); init_lerp = 0.1, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)

    G_init = Replicas.init(4, L)
    # @views G_init.blocks[:, :, 1] = G_init_single.blocks[1:L, 1:L, 1]
    # @views G_init.blocks[:, :, 2] = G_init_single.blocks[(L+1):2L, 1:L, 1]
    @views G_init.blocks[:, :, 3] .= 0
    # @views G_init.blocks[:, :, 4] = G_init_single.blocks[(3L+1):4L, 1:L, 1]
    # G_init.blocks[:, :, 3] .= 0

    syk = SYKData(1, 1, q, 4, β)

    G_temp = Replicas.init(4, L)
    Σ_temp = Replicas.init(4, L)

    path = "data/paper/weighted/1R4a/"
    filename = "weight_1R4a" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
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
        saddle = WeightedReplicas.log_saddle(G_temp, Σ_temp, w, syk)
        pf_minus, pf_plus = WeightedReplicas.pfaffians(Σ_temp, syk)
        p_plus = pf_plus / (pf_plus + pf_minus)
        entropy = binary_entropy(w)

        df = DataFrame(weight = w, saddle = saddle, pf_minus = pf_minus, pf_plus = pf_plus, p_plus = p_plus, entropy = entropy)

        CSV.write(file, df, append=true)
    end

end


L = 1000
ws = collect(0:2:100) / 100

generate_1R4a_weight_data(collect(76:2:100) / 100, L, 1.0, 8; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)


# for β in [2.0, 5.0, 10.0, 20.0, 50.0]
#     for q in [2, 4, 6, 8]
#         generate_1R2_weight_data(ws, L, β, q; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
#     end
# end


# for β in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
#     for q in [2, 4, 6, 8]
#         df = CSV.File("data/weighted/1R4a_log2/weighted_1R4a_beta$(β)_q$(q)_L1000.csv") |> DataFrame
#         df[:, 2] ./= log(2, ℯ)
#         df[:, 6] ./= log(2, ℯ)
#         path = "data/weighted/1R4a/"
#         filename = "weighted_1R4a_beta$(β)_q$(q)_L1000.csv"
#         file = path * filename
#         !ispath(path) && mkpath(path)
#         custom_headers = ["weight", "saddle", "pf_minus", "pf_plus", "p_plus", "entropy"]
#         CSV.write(file, df, header=custom_headers)
#     end
# end
