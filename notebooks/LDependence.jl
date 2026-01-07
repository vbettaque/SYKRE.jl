using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas
using SYKRE.TFDReplicas

using Plots, DataFrames, CSV


function generate_det_plus_dependence(Ls, q, β; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    syk = SYKData(1, 1, q, 1, β)

    path = "data/L_dependence/"
    filename = "dets" * "_beta" * string(β) * "_q" * string(q) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "L,det_minus,det_plus,det_plus_approx\n")
    end

    for i in eachindex(Ls)
        L = Ls[i]
        @info "$(i) out of $(length(Ls)): L = $(L)"

        G_init = Replicas.init(1, L)

        _, Σ = WeightedReplicas.schwinger_dyson(G_init, 0, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        pf_minus, pf_plus = WeightedReplicas.pfaffians(Σ, syk)
        det_plus_approx = abs((β/L)^2 * sum(Σ.blocks)) 

        df = DataFrame(L = L, det_minus = pf_minus^2, det_plus = pf_plus^2, det_plus_approx = det_plus_approx)

        CSV.write(file, df, append=true)
    end
end

Ls = 100:100:2000
generate_det_plus_dependence(Ls, 4, 20; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)



# data_5_4 = CSV.File("data/L_dependence/pfaffians_beta5_q4.csv") |> DataFrame
# data_10_4 = CSV.File("data/L_dependence/dets_beta10_q4.csv") |> DataFrame
# data_20_4 = CSV.File("data/L_dependence/pfaffians_beta20_q4.csv") |> DataFrame

# xs = LinRange(50, 2000, 2000)

# p = plot(data_10_4[:, 1], data_10_4[:, 3], label="pf_plus")
# plot!(data_10_4[:, 1], data_10_4[:, 4], label="pf_plus_approx")
# plot!(xs, 55 ./ xs .+ 0.015, label="55 / L + 0.015")
# plot!(xs, 120 ./ xs .+ 0.040, label="120 / L + 0.040")
# ylims!(0, 1) 
# xaxis!("1/L")
# display(p)



# plot(1 ./ data_20_4[:, 1], data_20_4[:, 2])
# plot!(1 ./ data_20_4[:, 1], data_20_4[:, 3])



data_1_4 = CSV.File("data/L_dependence/dets_beta1_q4.csv") |> DataFrame

xs = LinRange(50, 2000, 2000)

p = plot(data_1_4[:, 1], data_1_4[:, 3], label="pf_plus")
plot!(data_1_4[:, 1], data_1_4[:, 3], label="pf_plus_approx")
plot!(xs, 0.115 ./ xs, label="0.115 / L")
ylims!(0, 0.0012)
xaxis!("1/L")
title!("beta = 1, q = 4")
display(p)



# plot(1 ./ data_20_4[:, 1], data_20_4[:, 2])
# plot!(1 ./ data_20_4[:, 1], data_20_4[:, 3])