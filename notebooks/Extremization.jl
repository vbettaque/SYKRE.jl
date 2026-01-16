using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas
using SYKRE.TFDReplicas

using CairoMakie, DataFrames, CSV, LsqFit, Printf, Latexify, LaTeXStrings, Colors

invphi = (sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - sqrt(5)) / 2

function binary_entropy(w)
    @assert 0 ≤ w ≤ 1
    if iszero(w) || isone(w)
        return 0
    end
    return - w * log(w) - (1 - w) * log(1 - w)
end

function gss(f, a, b, tol=1e-5)
    a, b = min(a, b), max(a, b)
    h = b - a
    h <= tol && return (a + b) / 2

    # Required steps to achieve tolerance
    n = Int(ceil(log(tol / h) / log(invphi)))
    @info "$(n) steps required"

    c, d = a + invphi2 * h, a + invphi * h
    yc, yd = f(c), f(d)
    @info "c = $(c), d = $(d)"
    @info "yc = $(yc), yd = $(yd)"
    for i in 1:(n - 1)
        @info "GSS Iteration $(i)"
        h *= invphi
        if yc < yd
            b, d = d, c
            yd = yc
            c = a + invphi2 * h
            yc = f(c)
            @info "c = $(c), d = $(d)"
            @info "yc = $(yc), yd = $(yd)"
        else
            a, c = c, d
            yc = yd
            d = a + invphi * h
            yd = f(d)
            @info "c = $(c), d = $(d)"
            @info "yc = $(yc), yd = $(yd)"
        end
    end

    if yc < yd
        return (a + d) / 2
    else
        return (b + c) / 2
    end
end

function generate_extremized_purity_data(βs, q, L; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    path = "data/purity/extremized/"
    filename = "purity" * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "beta,log_purity,w_crit,log_Z\n")
    end

    G_init_2 = Replicas.init(2, L)
    G_init_Z = Replicas.init(1, L)
    for i = eachindex(βs)
        β = βs[i]
        @info "$(i) out of $(length(βs)): β = $(β)"

        @info "Computing log_Z"
        syk_2 = SYKData(1, 1, q, 2, β)
        syk_Z = SYKData(1, 1, q, 1, β)
        G_Z, Σ_Z = WeightedReplicas.schwinger_dyson(G_init_Z, 0, syk_Z; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        log_Z = WeightedReplicas.log_saddle(G_Z, Σ_Z, 0, syk_Z)
        @info "log_Z = $(log_Z)"

        @info "Extremizing H(w) - I(w)"
        function f(w)
            G_2, Σ_2 = WeightedReplicas.schwinger_dyson(G_init_2, w, syk_2; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
            return -(WeightedReplicas.log_saddle(G_2, Σ_2, w, syk_2) + binary_entropy(w) - 2 * log_Z - log(2)/2)
        end
        w_crit = gss(f, 0.0, 0.5, 1e-3)

        @info "Computing log_purity at w_crit"
        log_purity = f(w_crit)

        df = DataFrame(beta = β, log_purity = log_purity, w_crit = w_crit, log_Z = log_Z)

        CSV.write(file, df, append=true)
    end
end

function generate_ordinary_purity_data(βs, q, L; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    path = "data/purity/ordinary/"
    filename = "purity" * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "beta,log_purity,w_crit,log_Z\n")
    end

    G_init_2 = Replicas.init(1, 2*L)
    G_init_Z = Replicas.init(1, L)
    for i = eachindex(βs)
        β = βs[i]
        @info "$(i) out of $(length(βs)): β = $(β)"

        @info "Computing log_Z"
        syk_Z = SYKData(1, 1, q, 1, β)
        G_Z, Σ_Z = WeightedReplicas.schwinger_dyson(G_init_Z, 0, syk_Z; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        log_Z = WeightedReplicas.log_saddle(G_Z, Σ_Z, 0, syk_Z)
        @info "log_Z = $(log_Z)"

        @info "Computing log_purity"
        syk_2 = SYKData(1, 1, q, 1, 2*β)
        G_2, Σ_2 = WeightedReplicas.schwinger_dyson(G_init_2, 0, syk_2; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        log_Z_2 = WeightedReplicas.log_saddle(G_2, Σ_2, 0, syk_2)
        log_purity = -(log_Z_2 - 2 * log_Z)
        @info "log_purity = $(log_purity)"

        @info "Computing w_crit"
        Σ_2_rep = Replicas.init(2, L)
        Σ_2_rep.blocks[:, :, 1] = Σ_2.blocks[1:L, 1:L, 1]
        Σ_2_rep.blocks[:, :, 2] = Σ_2.blocks[L+1:2L, 1:L, 1]
        pf_minus, pf_plus = WeightedReplicas.pfaffians(Σ_2_rep, SYKData(1, 1, q, 2, β))
        w_crit = pf_plus / (pf_minus + pf_plus)
        @info "w_crit = $(w_crit)"

        df = DataFrame(beta = β, log_purity = log_purity, w_crit = w_crit, log_Z = log_Z)

        CSV.write(file, df, append=true)
    end
end

function generate_purity_data_from_w_crit(q, L; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    path = "data/purity/w_crit/"
    filename = "purity" * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "beta,log_purity,w_crit,log_Z\n")
    end

    G_init_2 = Replicas.init(2, L)

    data = CSV.File("data/purity/ordinary/purity_q$(q)_L$(L).csv") |> DataFrame
    for i = 1:length(data[:, 1])
        β = data[i, 1]
        @info "$(i) out of $(length(βs)): β = $(β)"

        @info "Getting log_Z"
        log_Z = data[i, 4]
        @info "log_Z = $(log_Z)"

        @info "Computing log_purity"
        syk_2 = SYKData(1, 1, q, 2, β)
        w_crit = data[i,3]
        G_2, Σ_2 = WeightedReplicas.schwinger_dyson(G_init_2, w_crit, syk_2; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        log_purity = -(WeightedReplicas.log_saddle(G_2, Σ_2, w_crit, syk_2) + binary_entropy(w_crit) - 2 * log_Z - log(2)/2)
        @info "log_purity = $(log_purity)"

        df = DataFrame(beta = β, log_purity = log_purity, w_crit = w_crit, log_Z = log_Z)

        CSV.write(file, df, append=true)
    end
end

L = 1000
βs = range(0.5, step=0.5, stop=20)

# generate_extremized_purity_data(βs, 6, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
generate_extremized_purity_data(βs, 8, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
# generate_ordinary_purity_data(βs, q, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)

# generate_purity_data_from_w_crit(4, 1000; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
