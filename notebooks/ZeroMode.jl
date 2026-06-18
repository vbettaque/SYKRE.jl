using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays, Combinatorics, BenchmarkTools, Profile, CairoMakie, LsqFit

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


function generate_zero_mode_data(ws, L, β, q; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    G_init = Replicas.init(2, L)

    syk = SYKData(1, 1, q, 2, β)

    G_temp = Replicas.init(2, L)
    Σ_temp = Replicas.init(2, L)

    path = "data/zero_mode/"
    filename = "zero_mode" * "_beta" * string(β) * "_q" * string(q) * "_L" * string(L) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "weight,G_0,Sigma_0\n")
    end

    for i in eachindex(ws)
        w = ws[i]
        @info "$(i) out of $(length(ws)): w = $(w)"

        G_0 = 0
        Σ_0 = 0

        if !iszero(w)
            G_temp, Σ_temp = WeightedReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
            # @info G_temp.blocks[:, :, 2]
            G_0 = sum(G_temp.blocks[:, :, 2]) / L^2
            Σ_0 = sum(Σ_temp.blocks[:, :, 2]) / L^2
        end

        df = DataFrame(weight = w, G_0 = G_0, Σ_0 = Σ_0)

        Replicas.plot(G_temp, title = "β = $(β), w = $(w)")

        CSV.write(file, df, append=true)
    end

end

β = 20
q = 2
L = 1000
ws = 0:0.02:1

generate_zero_mode_data(ws, L, β, q; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)



# fit_model(w, p) = (w ./ (p[1]^2)).^(1 / 4)
# p0 = [5.]

# data = CSV.File("data/zero_mode/zero_mode_beta20_q4_L1000.csv") |> DataFrame

# fit = curve_fit(fit_model, data[1:end, 1], data[1:end, 2], p0)
# p = fit.param

# f = Figure()
# ax = Axis(f[1, 1])
# lines!(ax, data[:, 1], data[:, 2])
# lines!(ax, data[:, 1], fit_model(data[:, 1], [20]))
# f




