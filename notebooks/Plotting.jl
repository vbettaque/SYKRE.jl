using CSV, DataFrames, Statistics, CairoMakie, LinearAlgebra, FFTW, Latexify, BlockArrays, Colors


inch = 96
pt = 4/3
cm = inch / 2.54

invphi = (sqrt(5) - 1) / 2

width_large = 6.5inch
height_large = invphi * width_large

width_small = 3.25inch
height_small = invphi * width_small

font_large = 12pt
font_small = 10pt

legend_color = RGBA(1, 1, 1, 0.8)

function plot_R2_weighted(β, qs, L, save_fig = false)
    mkpath("figures/weighted/R2/")

    width = width_small
    height = height_small
    font = font_small

    weighted_data = []
    Z_data = []
    for q in qs
        weighted_df = CSV.File("data/weighted/1R2/weighted_1R2_beta$(β)_q$(q)_L$(L).csv") |> DataFrame
        Z_df = CSV.File("data/partition_function/partition_function_q$(q)_L$(L).csv") |> DataFrame
        Z_df = Z_df[Z_df.beta .== β, :][1, :]
        weighted_df[1, 2] = 2 * Z_df[2]
        weighted_df[1, 3] = Z_df[3]^2
        weighted_df[1, 4] = Z_df[4]^2
        weighted_df[1, 5] = weighted_df[1, 4] / (weighted_df[1, 3] + weighted_df[1, 4])
        push!(weighted_data, weighted_df)
        push!(Z_data, Z_df)
    end

    # -I(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$w$",
            ylabel = L"$\ln\left(\overline{\xi(w)^2}\right) / N$",
        )
        for i in eachindex(qs)
            lines!(
                ax,
                weighted_data[i][:,1],
                weighted_data[i][:,2] .- 2 * Z_data[i][2],
                color = Makie.wong_colors()[qs[i]÷2],
                label= L"$q = %$(qs[i])$"
            )
        end
        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0), backgroundcolor = legend_color)
        display(f)
        save_fig && save("figures/weighted/R2/weighted_action_R2_beta$(β)_L$(L).pdf", f)
    end

    # -I(w) + H(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$w$",
            ylabel = L"$\ln\left(\overline{\xi(w)^2}\right) / N + H(w)$",
        )
        for i in eachindex(qs)
            lines!(
                ax,
                weighted_data[i][:,1],
                weighted_data[i][:,2] .- 2 * Z_data[i][2] + weighted_data[i][:,6],
                color = Makie.wong_colors()[qs[i]÷2],
                label= L"$q = %$(qs[i])$"
            )
        end
        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0), backgroundcolor = legend_color)
        display(f)
        save_fig && save("figures/weighted/R2/weighted_action_entropy_R2_beta$(β)_L$(L).pdf", f)
    end

    # w_crit

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$w$",
            ylabel = L"$w_\mathrm{crit}$",
        )
        for i in eachindex(qs)
            lines!(
                ax,
                weighted_data[i][:,1],
                weighted_data[i][:,5],
                color = Makie.wong_colors()[qs[i]÷2],
                label= L"$q = %$(qs[i])$"
            )
        end
        lines!(
            ax,
            LinRange(0, 1, 100),
            LinRange(0, 1, 100),
            color = :gray,
            linestyle = :dash,
            label= L"$w$"
        )
        axislegend(position = :lt, rowgap = -5, padding = (6, 6, 0, 0), backgroundcolor = legend_color)
        max_p = maximum(weighted_data[i][end,5] for i in eachindex(qs))
        ylims!(-0.05 * max_p, 1.05 * max_p)
        display(f)
        save_fig && save("figures/weighted/R2/weighted_w_crit_R2_beta$(β)_L$(L).pdf", f)
    end
end

# function plot_1R2_comparison(β, q, L, save=false)

#     data = CSV.File("data/weighted/1R2/weighted_1R2_beta$(β)_q$(q)_L$(L).csv") |> DataFrame
#     Z_data = CSV.File("data/partition_function/partition_function_q$(q)_L$(L).csv") |> DataFrame
#     Z_data = Z_data[Z_data.beta .== β, :][1, :]


#     # -I(w)

#     ws = LinRange(0, 1, 1000)
#     saddle_approx = ws .* 2 .* log(Z_data[4]) + (1 .- ws) .* 2 .* log(Z_data[3]) .- 2 .* Z_data[5]

#     p = plot()
#     plot!(data[:,1], data[:,2], label="actual")
#     plot!(ws, saddle_approx, label="approximation")

#     xlabel!("\$w\$")
#     ylabel!("\$-I(w)\$")

#     title!("1R2 (\$ \\beta = $(β), q = $(q)\$)")

#     display(p)
#     save && savefig("figures/weighted/1R2/weighted_action_1R2_beta$(β)_L$(L).pdf")
# end

function plot_R4_weighted_single_q(β, q, L, plot_2R2 = true, plot_1R4a = true, plot_1R4b = true, save_fig=false)
    mkpath("figures/weighted/R4/single_q/q$(q)")

    width = width_small
    height = height_small
    font = font_small

    data_Z = CSV.File("data/partition_function/partition_function_q$(q)_L$(L).csv") |> DataFrame
    data_Z = data_Z[data_Z.beta .== β, :][1, :]

    data_2R2 = CSV.File("data/weighted/1R2/weighted_1R2_beta$(β)_q$(q)_L$(L).csv") |> DataFrame
    data_2R2[:, 2] = 2 * data_2R2[:, 2]
    data_2R2[:, 3] = data_2R2[:, 3].^2
    data_2R2[:, 4] = data_2R2[:, 4].^2
    data_2R2[:, 5] = data_2R2[:, 4] ./ (data_2R2[:, 3] .+ data_2R2[:, 4])
    data_2R2[1, 2] = 4 * data_Z[2]
    data_2R2[1, 3] = data_Z[3]^4
    data_2R2[1, 4] = data_Z[4]^4
    data_2R2[1, 5] = data_2R2[1, 4] / (data_2R2[1, 3] + data_2R2[1, 4])

    data_1R4a = CSV.File("data/weighted/1R4a/weighted_1R4a_beta$(β)_q$(q)_L$(L).csv") |> DataFrame
    data_1R4a[1, 2] = 4 * data_Z[2]
    data_1R4a[1, 3] = data_Z[3]^4
    data_1R4a[1, 4] = data_Z[4]^4
    data_1R4a[1, 5] = data_1R4a[1, 4] / (data_1R4a[1, 3] + data_1R4a[1, 4])

    data_1R4b = CSV.File("data/weighted/1R4b/weighted_1R4b_beta$(β)_q$(q)_L$(L).csv") |> DataFrame
    data_1R4b[1, 2] = 4 * data_Z[2]
    data_1R4b[1, 3] = data_Z[3]^4
    data_1R4b[1, 4] = data_Z[4]^4
    data_1R4b[1, 5] = data_1R4b[1, 4] / (data_1R4b[1, 3] + data_1R4b[1, 4])

    # -I(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"$w$",
            ylabel = L"$\ln\left(\overline{\xi(w)^4}\right) / N$",
        )
        plot_2R2 && lines!(
            ax,
            data_2R2[:,1],
            data_2R2[:,2] .- 4 * data_Z[2],
            color = Makie.wong_colors()[1],
            label= "2R2"
        )
        plot_1R4a && lines!(
            ax,
            data_1R4a[:,1],
            data_1R4a[:,2] .- 4 * data_Z[2],
            color = Makie.wong_colors()[2],
            label= "1R4"
        )
        plot_1R4b && lines!(
            ax,
            data_1R4b[:,1],
            data_1R4b[:,2] .- 4 * data_Z[2],
            color = Makie.wong_colors()[3],
            label= "1R4b"
        )

        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0), backgroundcolor = legend_color)
        display(f)
        save_fig && save("figures/weighted/R4/single_q/q$(q)/weighted_action_R4_beta$(β)_q$(q)_L$(L).pdf", f)
    end

    # # -I(w) + H(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"$w$",
            ylabel = L"$\ln\left(\overline{\xi(w)^4}\right) / N + H(w)$",
        )
        plot_2R2 && lines!(
            ax,
            data_2R2[:,1],
            data_2R2[:,2] .- 4 * data_Z[2] + data_2R2[:,6],
            color = Makie.wong_colors()[1],
            label= "2R2"
        )
        plot_1R4a && lines!(
            ax,
            data_1R4a[:,1],
            data_1R4a[:,2] .- 4 * data_Z[2] + data_1R4a[:,6],
            color = Makie.wong_colors()[2],
            label= "1R4"
        )
        plot_1R4b && lines!(
            ax,
            data_1R4b[:,1],
            data_1R4b[:,2] .- 4 * data_Z[2] + data_1R4b[:,6],
            color = Makie.wong_colors()[3],
            label= "1R4b"
        )

        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0))
        display(f)
        save_fig && save("figures/weighted/R4/single_q/q$(q)/weighted_action_entropy_R4_beta$(β)_q$(q)_L$(L).pdf", f)
    end

    # w_crit

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"$w$",
            ylabel = L"$w_\mathrm{crit}$",
        )
        plot_2R2 && lines!(
            ax,
            data_2R2[:,1],
            data_2R2[:,5],
            color = Makie.wong_colors()[1],
            label= "2R2"
        )
        plot_1R4a && lines!(
            ax,
            data_1R4a[:,1],
            data_1R4a[:,5],
            color = Makie.wong_colors()[2],
            label= "1R4"
        )
        plot_1R4b && lines!(
            ax,
            data_1R4b[:,1],
            data_1R4b[:,5],
            color = Makie.wong_colors()[3],
            label= "1R4b"
        )
        lines!(
            ax,
            LinRange(0, 1, 100),
            LinRange(0, 1, 100),
            color = :gray,
            linestyle = :dash,
            label= L"$w$"
        )
        axislegend(position = :lt, rowgap = -5, padding = (6, 6, 0, 0))
        max_p = max(data_2R2[end,5], data_1R4a[end,5], data_1R4b[end,5])
        ylims!(-0.05 * max_p, 1.05 * max_p)
        display(f)
        save_fig && save("figures/weighted/R4/single_q/q$(q)/weighted_w_crit_R4_beta$(β)_q$(q)_L$(L).pdf", f)
    end
end


function plot_2R2_weighted(β, qs, L, save_fig=false)
    mkpath("figures/weighted/R4/2R2")

    width = width_small
    height = height_small
    font = font_small

    data_2R2 = []
    data_Z = []

    for i = eachindex(qs)
        q = qs[i]
        data_Z_q = CSV.File("data/partition_function/partition_function_q$(q)_L$(L).csv") |> DataFrame
        data_Z_q = data_Z_q[data_Z_q.beta .== β, :][1, :]
        push!(data_Z, data_Z_q)

        data_2R2_q = CSV.File("data/weighted/1R2/weighted_1R2_beta$(β)_q$(q)_L$(L).csv") |> DataFrame
        data_2R2_q[:, 2] = 2 * data_2R2_q[:, 2]
        data_2R2_q[:, 3] = data_2R2_q[:, 3].^2
        data_2R2_q[:, 4] = data_2R2_q[:, 4].^2
        data_2R2_q[:, 5] = data_2R2_q[:, 4] ./ (data_2R2_q[:, 3] .+ data_2R2_q[:, 4])
        data_2R2_q[1, 2] = 4 * data_Z_q[2]
        data_2R2_q[1, 3] = data_Z_q[3]^4
        data_2R2_q[1, 4] = data_Z_q[4]^4
        data_2R2_q[1, 5] = data_2R2_q[1, 4] / (data_2R2_q[1, 3] + data_2R2_q[1, 4])

        push!(data_2R2, data_2R2_q)
    end

    # display(data_Z)

    # -I(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"$w$",
            ylabel = L"$\ln\left(\overline{\xi(w)^4}\right) / N$",
        )

        for i = eachindex(qs)
            q = qs[i]
            lines!(
                ax,
                data_2R2[i][:,1],
                data_2R2[i][:,2] .- 4 * data_Z[i][2],
                color = Makie.wong_colors()[q ÷ 2],
                label= L"q = %$(q)"
            )
        end

        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0), backgroundcolor = legend_color)
        display(f)
        save_fig && save("figures/weighted/R4/2R2/weighted_action_2R2_beta$(β)_L$(L).pdf", f)
    end

    # # -I(w) + H(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"$w$",
            ylabel = L"$\ln\left(\overline{\xi(w)^4}\right) / N + H(w)$",
        )

        for i = eachindex(qs)
            q = qs[i]
            lines!(
                ax,
                data_2R2[i][:,1],
                data_2R2[i][:,2] .- 4 * data_Z[i][2] + data_2R2[i][:,6],
                color = Makie.wong_colors()[q ÷ 2],
                label= L"q = %$(q)"
            )
        end

        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0))
        display(f)
        save_fig && save("figures/weighted/R4/2R2/weighted_action_entropy_2R2_beta$(β)_L$(L).pdf", f)
    end

    # w_crit

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, 1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"$w$",
            ylabel = L"$w_\mathrm{crit}$",
        )

        for i = eachindex(qs)
            q = qs[i]
            lines!(
                ax,
                data_2R2[i][:,1],
                data_2R2[i][:,5],
                color = Makie.wong_colors()[q ÷ 2],
                label= L"q = %$(q)"
            )
        end

        lines!(
            ax,
            LinRange(0, 1, 100),
            LinRange(0, 1, 100),
            color = :gray,
            linestyle = :dash,
            label= L"$w$"
        )

        axislegend(position = :lt, rowgap = -5, padding = (6, 6, 0, 0), backgroundcolor = legend_color)
        max_p = maximum([data_2R2[i][end,5] for i = eachindex(qs)])
        ylims!(-0.05 * max_p, 1.05 * max_p)
        display(f)
        save_fig && save("figures/weighted/R4/2R2/weighted_w_crit_2R2_beta$(β)_L$(L).pdf", f)
    end
end


function plot_norm(qs, L, save_fig=false)
    mkpath("figures/norm_purity/")

    width = width_large
    height = height_large
    font = font_large

    data_norm = []
    data_error = []

    for i = eachindex(qs)
        q = qs[i]
        df_norm = CSV.File("data/norm/norm_q$(q)_L$(L).csv") |> DataFrame
        push!(data_norm, df_norm)

        df_purity_ordinary = CSV.File("data/purity/ordinary/purity_q$(q)_L$(L).csv") |> DataFrame
        df_purity_extremized = CSV.File("data/purity/extremized/purity_q$(q)_L$(L).csv") |> DataFrame
        push!(data_error, abs.(df_purity_extremized[:, 2] - df_purity_ordinary[:, 2]) ./ 2)
    end

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"β",
            ylabel = L"\ln\left(F_1(\rho)\right) / N - \ln(2)/2",
        )
        for i in eachindex(qs)
            q = qs[i]
            errorbars!(
                ax,
                data_norm[i][:,1],
                data_norm[i][:,2] .- log(2) / 2,
                data_error[i][:],
                data_error[i][:],
                whiskerwidth = 3,
                direction = :y
            )
            lines!(
                ax,
                data_norm[i][:,1],
                data_norm[i][:,2] .- log(2) / 2,
                color = Makie.wong_colors()[q ÷ 2],
                label= L"q = %$(q)"
            )
        end

        axislegend(position = :rb, rowgap = 0, padding = (7, 7, 2, 2), margin = (10, 10, 10, 10))
        display(f)
        save_fig && save("figures/norm_purity/norm_L$(L).pdf", f)
    end
end


function plot_purity(qs, L, save_fig=false)
    mkpath("figures/norm_purity/")

    width = width_large
    height = height_large
    font = font_large

    data_ordinary = []
    data_extremized = []

    for i = eachindex(qs)
        q = qs[i]
        df_ordinary = CSV.File("data/purity/ordinary/purity_q$(q)_L$(L).csv") |> DataFrame
        push!(data_ordinary, df_ordinary)
        df_extremized = CSV.File("data/purity/extremized/purity_q$(q)_L$(L).csv") |> DataFrame
        push!(data_extremized, df_extremized)
    end

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"β",
            ylabel = L"S_2(\rho) / N",
        )
        for i in eachindex(qs)
            q = qs[i]
            lines!(
                ax,
                data_ordinary[i][:,1],
                data_ordinary[i][:,2],
                color = Makie.wong_colors()[q ÷ 2],
                label= L"q = %$(q)"
            )
        end
        for i in eachindex(qs)
            q = qs[i]
            lines!(
                ax,
                data_extremized[i][:,1],
                data_extremized[i][:,2],
                color = Makie.wong_colors()[q ÷ 2],
                label= L"q = %$(q)",
                linestyle = :dash,
            )
        end

        group_ordinary = [LineElement(color = Makie.wong_colors()[q ÷ 2], linestyle = nothing) for q in qs]
        group_extremized = [LineElement(color = Makie.wong_colors()[q ÷ 2], linestyle = :dash) for q in qs]

        Legend(
            f[1,1],
            [group_ordinary, group_extremized],
            [[L"q = %$(q)" for q in qs], [L"q = %$(q)" for q in qs]],
            [L"\ln(Z(2β)) / N", L"\max\left(H(w) - I^{(2, w)}_\mathrm{SYK}\right)"],
            nbanks = 2,
            tellheight = false,
            tellwidth = false,
            halign = :left,
            valign = :bottom,
            margin = (10, 10, 10, 10),
            fontsize = font,
            titlegap = 0,
            groupgap = 10,
            backgroundcolor = RGBA(1, 1, 1, 0.9)
        )

        # axislegend(position = :lt, rowgap = 0, padding = (7, 7, 2, 2), margin = (10, 10, 10, 10))
        display(f)
        save_fig && save("figures/norm_purity/purity_L$(L).pdf", f)
    end
end


function plot_norm_vs_purity(qs, L, save_fig=false)
    mkpath("figures/norm_purity/")

    width = width_large
    height = height_large
    font = font_large

    data_norm = []
    data_error = []
    data_purity_ordinary = []
    # data_purity_extremized = []

    for i = eachindex(qs)
        q = qs[i]
        df_norm = CSV.File("data/norm/norm_q$(q)_L$(L).csv") |> DataFrame
        push!(data_norm, df_norm)
        df_purity_ordinary = CSV.File("data/purity/ordinary/purity_q$(q)_L$(L).csv") |> DataFrame
        push!(data_purity_ordinary, df_purity_ordinary)
        df_purity_extremized = CSV.File("data/purity/extremized/purity_q$(q)_L$(L).csv") |> DataFrame
        push!(data_error, abs.(df_purity_extremized[:, 2] - df_purity_ordinary[:, 2]) ./ 2)
    end

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"β",
            # ylabel = L"$\ln(2)/4 - S_2(\rho) / 2N$",
        )
        for i in eachindex(qs)
            q = qs[i]
            errorbars!(
                ax,
                data_norm[i][:,1],
                data_norm[i][:,2] .- log(2) / 2,
                data_error[i][:],
                data_error[i][:],
                whiskerwidth = 3,
                direction = :y
            )
            lines!(
                ax,
                data_norm[i][:,1],
                data_norm[i][:,2] .- log(2) / 2,
                color = Makie.wong_colors()[q ÷ 2],
                # label= L"\ln(F_1(\rho)) / N - \ln(2)/2 \quad (q = %$(q))",
                label= L"\ln\left(\tilde{F}_1(\rho)\right) / N \quad (q = %$(q))",

            )
            lines!(
                ax,
                data_purity_ordinary[i][:,1],
                (log(2) / 2 .- data_purity_ordinary[i][:,2])/2,
                color = Makie.wong_colors()[q ÷ 2],
                linestyle = :dash,
                label= L"\tilde{S}_2(\rho) / N \quad (q = %$(q))"
            )
        end

        group_norm = [LineElement(color = Makie.wong_colors()[q ÷ 2], linestyle = nothing) for q in qs]
        group_purity = [LineElement(color = Makie.wong_colors()[q ÷ 2], linestyle = :dash) for q in qs]

        Legend(
            f[1,1],
            [group_norm, group_purity],
            [[L"q = %$(q)" for q in qs], [L"q = %$(q)" for q in qs]],
            [L"\ln(F_1(\rho)) / N - \ln(2)/2", L"\ln(2) / 4 - S_2(\rho) / 2N"],
            nbanks = length(qs),
            tellheight = false,
            tellwidth = false,
            halign = :right,
            valign = :bottom,
            margin = (10, 10, 10, 10),
            fontsize = font,
            titlegap = 0,
            groupgap = 10,
        )

        # axislegend(position = :rb, rowgap = 0, padding = (7, 7, 2, 2))
        display(f)
        save_fig && save("figures/norm_purity/norm_purity_L$(L).pdf", f)
    end
end

# plot_R2_weighted(0.1, [2, 4], 1000, true)
# plot_R2_weighted(0.2, [2, 4], 1000, true)
# plot_R2_weighted(0.5, [2, 4, 6, 8], 1000, true)
# plot_R2_weighted(1.0, [2, 4, 6, 8], 1000, true)
# plot_R2_weighted(2.0, [2, 4, 6, 8], 1000, true)
# plot_R2_weighted(5.0, [2, 4, 6, 8], 1000, true)
# plot_R2_weighted(10.0, [2, 4, 6, 8], 1000, true)
# plot_R2_weighted(20.0, [2, 4, 6, 8], 1000, true)
# plot_R2_weighted(50.0, [2, 4, 6, 8], 1000, true)


# for β in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
#     plot_R4_weighted_single_q(β, 4, 1000, true, true, false, true)
# end

plot_2R2_weighted(0.1, [2, 4], 1000, true)
plot_2R2_weighted(0.2, [2, 4], 1000, true)
for β in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    plot_2R2_weighted(β, [2, 4, 6, 8], 1000, true)
end



# plot_2R2_weighted(20.0, [2], 1000, true)

# plot_1_norm(50.0, [2, 4, 6, 8], 1000, false)


# plot_1_norm(20.0, [2, 4, 6, 8], 1000, false)

# plot_1R2_comparison(10., 4, 1000, false)

# plot_R4_weighted(50.0, 4, 1000, false)

# plot_norm([2, 4, 6, 8], 1000, true)
# plot_purity([2, 4, 6, 8], 1000, true)
# plot_norm_vs_purity([2, 4], L, true)
