using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas
using SYKRE.TFDReplicas

using CairoMakie, DataFrames, CSV, LsqFit, Printf, Latexify, LaTeXStrings

function generate_1R1_det_plus_dependence(Ls, q, β; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    syk = SYKData(1, 1, q, 1, β)

    path = "data/L_dependence/det_plus/1R1/"
    filename = "dets_1R1" * "_beta" * string(β) * "_q" * string(q) * ".csv"
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

function generate_1R2_det_plus_dependence(Ls, q, β; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    syk = SYKData(1, 1, q, 1, β)

    path = "data/L_dependence/det_plus/1R2/"
    filename = "dets_1R2" * "_beta" * string(β) * "_q" * string(q) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "L,det_minus,det_plus\n")
    end

    for i in eachindex(Ls)
        L = Ls[i]
        @info "$(i) out of $(length(Ls)): L = $(L)"

        G_init = Replicas.init(2, L)

        _, Σ = WeightedReplicas.schwinger_dyson(G_init, 0.5, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        pf_minus, pf_plus = WeightedReplicas.pfaffians(Σ, syk)

        df = DataFrame(L = L, det_minus = pf_minus^2, det_plus = pf_plus^2)

        CSV.write(file, df, append=true)
    end
end

function generate_jump_dependence(Ls, q, β, w; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    syk_1 = SYKData(1, 1, q, 1, β)
    syk_2 = SYKData(1, 1, q, 2, β)

    path = "data/L_dependence/jump/"
    filename = "jump_beta" * string(β) * "_w" * string(w) * "_q" * string(q) * ".csv"
    file = path * filename
    !ispath(path) && mkpath(path)
    if !isfile(file)
        touch(file)
        write(file, "L,saddle_0,saddle_w\n")
    end

    for i in eachindex(Ls)
        L = Ls[i]
        @info "$(i) out of $(length(Ls)): L = $(L)"

        G_init_1 = Replicas.init(1, L)
        G_init_2 = Replicas.init(2, L)

        G_1, Σ_1 = WeightedReplicas.schwinger_dyson(G_init_1, 0.0, syk_1; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        saddle_0 = 2 * WeightedReplicas.log_saddle(G_1, Σ_1, 0, syk_1)
        G_2, Σ_2 = WeightedReplicas.schwinger_dyson(G_init_2, w, syk_2; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
        saddle_w = WeightedReplicas.log_saddle(G_2, Σ_2, w, syk_2)

        df = DataFrame(L = L, saddle_0 = saddle_0, saddle_w = saddle_w)

        CSV.write(file, df, append=true)
    end
end

Ls = 100:100:2000
generate_jump_dependence(Ls, 4, 20, 0.01; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
# generate_1R1_det_plus_dependence(Ls, 2, 1; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
# generate_1R2_det_plus_dependence(Ls, 4, 20; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)


############## 1R1 Plots ##################


model(L, p) = p[1] ./ L.^p[2] .+ p[3]

data_1_4 = CSV.File("data/L_dependence/det_plus/1R1/dets_1R1_beta1_q4.csv") |> DataFrame
data_5_4 = CSV.File("data/L_dependence/det_plus/1R1/dets_1R1_beta5_q4.csv") |> DataFrame
data_10_4 = CSV.File("data/L_dependence/det_plus/1R1/dets_1R1_beta10_q4.csv") |> DataFrame
data_20_4 = CSV.File("data/L_dependence/det_plus/1R1/dets_1R1_beta20_q4.csv") |> DataFrame

Ls = LinRange(1, 2500, 2500)

inch = 96
pt = 4/3
cm = inch / 2.54

width = 3.25inch
height = 3 * width / 4
font = 10pt

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_1_4[:, 1], data_1_4[:, 3], p0)
    p_det = round.(fit_det.param, digits = 2)
    fit_approx = curve_fit(model, data_1_4[:, 1], data_1_4[:, 4], p0)
    p_approx = round.(fit_approx.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
            ylabelpadding = 0,
        )
        scatter!(
            ax,
            data_1_4[:, 1],
            data_1_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$|\det(\mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)|$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[1]) / L^{%$(p_det[2])}$"
        )
        scatter!(
            ax,
            data_1_4[:, 1],
            data_1_4[:, 4],
            color = Makie.wong_colors()[2],
            label=L"$(\Delta\tau)^2 \, | \mathrm{tr}(\mathbb{1} \, \Sigma) |$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_approx.param),
            color = Makie.wong_colors()[2],
            linestyle = :dash,
            label= L"$\approx %$(p_approx[1]) / L^{%$(p_approx[2])}$"
        )
        delta_y = data_1_4[1, 3]
        ylims!(ax, - 0.05 * delta_y, data_1_4[1, 3] + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rt)
        resize_to_layout!(f)
        save("figures/L_dependence/1R1/dets_1R1_beta1_q4.pdf", f)
        f
    end
end

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_5_4[14:end, 1], data_5_4[14:end, 3], p0)
    p_det = round.(fit_det.param, digits = 2)
    fit_approx = curve_fit(model, data_5_4[5:end, 1], data_5_4[5:end, 4], p0)
    p_approx = round.(fit_approx.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
        )
        scatter!(
            ax,
            data_5_4[:, 1],
            data_5_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$|\det(\mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)|$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[1]) / L^{%$(p_det[2])}$"
        )
        scatter!(
            ax,
            data_5_4[:, 1],
            data_5_4[:, 4],
            color = Makie.wong_colors()[2],
            label=L"$(\Delta\tau)^2 \, | \mathrm{tr}(\mathbb{1} \, \Sigma) |$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_approx.param),
            color = Makie.wong_colors()[2],
            linestyle = :dash,
            label= L"$\approx %$(p_approx[1]) / L^{%$(p_approx[2])}$"
        )
        delta_y = data_5_4[1, 4]
        ylims!(ax, - 0.05 * delta_y, data_5_4[1, 4] + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rt)
        resize_to_layout!(f)
        save("figures/L_dependence/1R1/dets_1R1_beta5_q4.pdf", f)
        f
    end
end

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_10_4[5:end, 1], data_10_4[5:end, 3], p0)
    p_det = round.(fit_det.param, digits = 2)
    fit_approx = curve_fit(model, data_10_4[5:end, 1], data_10_4[5:end, 4], p0)
    p_approx = round.(fit_approx.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
        )
        scatter!(
            ax,
            data_10_4[:, 1],
            data_10_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$|\det(\mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)|$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[1]) / L^{%$(p_det[2])}$"
        )
        scatter!(
            ax,
            data_10_4[:, 1],
            data_10_4[:, 4],
            color = Makie.wong_colors()[2],
            label=L"$(\Delta\tau)^2 \, | \mathrm{tr}(\mathbb{1} \, \Sigma) |$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_approx.param),
            color = Makie.wong_colors()[2],
            linestyle = :dash,
            label= L"$\approx %$(p_approx[1]) / L^{%$(p_approx[2])}$"
        )
        delta_y = data_10_4[1, 3]
        ylims!(ax, - 0.05 * delta_y, data_10_4[1, 3] + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rt)
        resize_to_layout!(f)
        save("figures/L_dependence/1R1/dets_1R1_beta10_q4.pdf", f)
        f
    end
end

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_20_4[5:end, 1], data_20_4[5:end, 3], p0)
    p_det = round.(fit_det.param, digits = 2)
    fit_approx = curve_fit(model, data_20_4[5:end, 1], data_20_4[5:end, 4], p0)
    p_approx = round.(fit_approx.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
        )
        scatter!(
            ax,
            data_20_4[:, 1],
            data_20_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$|\det(\mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)|$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[1]) / L^{%$(p_det[2])} %$(p_det[3])$"
        )
        scatter!(
            ax,
            data_20_4[:, 1],
            data_20_4[:, 4],
            color = Makie.wong_colors()[2],
            label=L"$(\Delta\tau)^2 \, | \mathrm{tr}(\mathbb{1} \, \Sigma) |$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_approx.param),
            color = Makie.wong_colors()[2],
            linestyle = :dash,
            label= L"$\approx %$(p_approx[1]) / L^{%$(p_approx[2])} %$(p_approx[3])$"
        )
        delta_y = data_20_4[1, 3]
        ylims!(ax, - 0.05 * delta_y, data_20_4[1, 3] + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rt)
        resize_to_layout!(f)
        save("figures/L_dependence/1R1/dets_1R1_beta20_q4.pdf", f)
        f
    end
end


############## 1R2 Plots ##################

model(L, p) = p[1] ./ L.^p[2] .+ p[3]

data_1_4 = CSV.File("data/L_dependence/det_plus/1R2/dets_1R2_beta1_q4.csv") |> DataFrame
data_5_4 = CSV.File("data/L_dependence/det_plus/1R2/dets_1R2_beta5_q4.csv") |> DataFrame
data_10_4 = CSV.File("data/L_dependence/det_plus/1R2/dets_1R2_beta10_q4.csv") |> DataFrame
data_20_4 = CSV.File("data/L_dependence/det_plus/1R2/dets_1R2_beta20_q4.csv") |> DataFrame

Ls = LinRange(1, 2500, 2500)

inch = 96
pt = 4/3
cm = inch / 2.54

width = 3.25inch
height = 3 * width / 4
font = 10pt

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_1_4[5:end, 1], data_1_4[5:end, 3], p0)
    p_det = round.(fit_det.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
            # yticks = [0.3],
        )
        scatter!(
            ax,
            data_1_4[:, 1],
            data_1_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$\det(I_2 \otimes \mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[3])$"
        )
        delta_y = 0.5
        ylims!(ax, -0.05 * delta_y, 0.5 + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rb)
        resize_to_layout!(f)
        save("figures/L_dependence/1R2/dets_1R2_beta1_q4.pdf", f)
        f
    end
end

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_5_4[5:end, 1], data_5_4[5:end, 3], p0)
    p_det = round.(fit_det.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
        )
        scatter!(
            ax,
            data_5_4[:, 1],
            data_5_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$\det(I_2 \otimes \mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[1]) / L^{%$(p_det[2])} + %$(p_det[3])$"
        )
        delta_y  = 5
        ylims!(ax, - 0.05 * delta_y, 5 + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rb)
        resize_to_layout!(f)
        save("figures/L_dependence/1R2/dets_1R2_beta5_q4.pdf", f)
        f
    end
end

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_10_4[5:end, 1], data_10_4[5:end, 3], p0)
    p_det = round.(fit_det.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
        )
        scatter!(
            ax,
            data_10_4[:, 1],
            data_10_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$\det(I_2 \otimes \mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[1]) / L^{%$(p_det[2])} + %$(p_det[3])$"
        )
        delta_y  = 50
        ylims!(ax, - 0.05 * delta_y, 50 + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rb)
        resize_to_layout!(f)
        save("figures/L_dependence/1R2/dets_1R2_beta10_q4.pdf", f)
        f
    end
end

begin
    p0 = [1., 1., 0.]
    fit_det = curve_fit(model, data_20_4[5:end, 1], data_20_4[5:end, 3], p0)
    p_det = round.(fit_det.param, digits = 2)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$L$",
            # ylabel = "Value",
        )
        scatter!(
            ax,
            data_20_4[:, 1],
            data_20_4[:, 3],
            color = Makie.wong_colors()[1],
            label=L"$\det(I_2 \otimes \mathcal{D}^+ \! - (\Delta \tau)^2 \, \Sigma)$"
        )
        lines!(
            ax,
            Ls,
            model(Ls, fit_det.param),
            color = Makie.wong_colors()[1],
            linestyle = :dash,
            label= L"$\approx %$(p_det[1]) / L^{%$(p_det[2])} + %$(p_det[3])$"
        )
        delta_y  = 5000
        ylims!(ax, - 0.05 * delta_y, 5000 + 0.05 * delta_y)
        xlims!(ax, -0.05 * 2000, 1.05 * 2000)
        axislegend(position = :rb)
        resize_to_layout!(f)
        save("figures/L_dependence/1R2/dets_1R2_beta20_q4.pdf", f)
        f
    end
end
