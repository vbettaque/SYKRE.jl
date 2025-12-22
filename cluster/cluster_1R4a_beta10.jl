#!/usr/bin/env julia

#SBATCH --job-name=1R4ab10
##SBATCH -p <list of partition names>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vbettaque@brandeis.edu

import Pkg
Pkg.add(path="..")

using Distributed


L = 1000
ws = collect(0:2:100) / 100
β = 10.0
qs = [2, 4, 6, 8]

addprocs(4; exeflags="--project")

@everywhere using Logging, CSV, DataFrames

@everywhere using SYKRE
@everywhere using SYKRE.SYK
@everywhere using SYKRE.Replicas
@everywhere using SYKRE.WeightedReplicas

@everywhere Logging.disable_logging(Logging.Info)

@everywhere function binary_entropy(w)
    @assert 0 ≤ w ≤ 1
    if iszero(w) || isone(w)
        return 0
    end
    return - w * log2(w) - (1 - w) * log2(1 - w)
end

@everywhere function generate_2R1_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
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
        @logmsg Logging.LogLevel(1) "$(i) out of $(length(ws)): w = $(w)"

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

@everywhere function generate_1R2_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
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
        @logmsg Logging.LogLevel(1) "$(i) out of $(length(ws)): w = $(w)"

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

@everywhere function generate_1R4a_weight_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
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
        @logmsg Logging.LogLevel(1) "$(i) out of $(length(ws)): w = $(w)"

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

        CSV.write(file, df, append=true)
    end

end

@sync @distributed for i in 1:4
    remotecall_fetch(generate_1R4a_weight_data, i, ws, L, β, qs[i]; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
end

for i in workers()
    rmprocs(i)
end
