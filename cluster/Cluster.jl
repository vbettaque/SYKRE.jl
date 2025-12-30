using CSV, DataFrames

using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas


function binary_entropy(w)
    @assert 0 ≤ w ≤ 1
    if iszero(w) || isone(w)
        return 0
    end
    return - w * log(w) - (1 - w) * log(1 - w)
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
        @info "$(i) out of $(length(ws)): w = $(w)"

        saddle = 0
        pf_minus = 0
        pf_plus = 0
        p_plus = 0
        entropy = 0

        G_temp, Σ_temp = WeightedReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        saddle = 2 * WeightedReplicas.log_saddle(G_temp, Σ_temp, 0, syk)
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
    filename = "weighted_1R2" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
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

        # Replicas.plot(G_temp, title = "β = $(β), w = $(w)")

        CSV.write(file, df, append=true)
    end

end


function generate_1R4a_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    G_init = Replicas.init(4, L)
    @views G_init.blocks[:, :, 3] .= 0

    syk = SYKData(1, 1, q, 4, β)

    G_temp = Replicas.init(4, L)
    Σ_temp = Replicas.init(4, L)

    path = "data/weighted/1R4a/"
    filename = "weighted_1R4a" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
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

function generate_1R4b_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    G_init = Replicas.init(4, L)

    syk = SYKData(1, 1, q, 4, β)

    G_temp = Replicas.init(4, L)
    Σ_temp = Replicas.init(4, L)

    path = "data/weighted/1R4b/"
    filename = "weighted_1R4b" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
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


function (@main)(args)
    rep = args[1]
    w_min = parse(Float64, args[2])
    w_max = parse(Float64, args[3])
    w_step = parse(Float64, args[4])
    β = parse(Float64, args[5])
    q = parse(Int, args[6])
    L = parse(Int, args[7])
    lerp = parse(Float64, args[8])

    ws = range(w_min, step=w_step, stop=w_max)
    if rep == "1R2"
        generate_1R2_weight_data(ws, L, β, q; init_lerp = lerp, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    elseif rep == "1R4a"
        generate_1R4a_weight_data(ws, L, β, q; init_lerp = lerp, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    elseif rep == "1R4b"
        generate_1R4b_weight_data(ws, L, β, q; init_lerp = lerp, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    else
        @error "Replica structure $(rep) not found."
    end
end

@isdefined(var"@main") ? (@main) : exit(main(ARGS))
