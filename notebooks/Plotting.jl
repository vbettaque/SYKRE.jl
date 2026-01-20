using CSV, DataFrames, Statistics, CairoMakie, LinearAlgebra, FFTW, Latexify, BlockArrays, Colors


function plot_1R2_weighted(β, qs, L, save_fig = false)
    inch = 96
    pt = 4/3
    cm = inch / 2.54

    width = 3.25inch
    height = 3 * width / 4
    font = 10pt

    mkpath("figures/weighted/1R2/")

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
        f = Figure(size = (width, height), figure_padding = (2, 1, -1, 1), fontsize = font)
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
        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0))
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/weighted/1R2/weighted_action_1R2_beta$(β)_L$(L).pdf", f)
    end

    # -I(w) + H(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (2, 1, -1, 1), fontsize = font)
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
        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0))
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/weighted/1R2/weighted_action_entropy_1R2_beta$(β)_L$(L).pdf", f)
    end

    # p_plus

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (1, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = "Experimental data and exponential fit",
            xlabel = L"$w$",
            ylabel = L"$p_+$",
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
        axislegend(position = :lt, rowgap = -5, padding = (6, 6, 0, 0))
        max_p = maximum(weighted_data[i][end,5] for i in eachindex(qs))
        ylims!(-0.05 * max_p, 1.05 * max_p)
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/weighted/1R2/weighted_p_plus_1R2_beta$(β)_L$(L).pdf", f)
    end

    # p = plot()

    # for i in eachindex(qs)
    #     plot!(data[i][:,1], data[i][:,5], label="\$q = $(qs[i])\$")
    # end

    # plot!(data[1][:,1], data[1][:,1], color=:grey, label="\$w\$")

    # xlabel!("\$w\$")
    # ylabel!("\$p_+\$")

    # title!("1R2 (\$ \\beta = $(β)\$)")

    # display(p)
    # save && savefig("figures/weighted/1R2/weighted_p_plus_1R2_beta$(β)_L$(L).pdf")
end

function plot_1_norm(β, qs, L, save=false)
    mkpath("figures/weighted/1R2/")

    Zs = []
    data = []
    for q in qs
        push!(data, CSV.File("data/weighted/1R2/weighted_1R2_beta$(β)_q$(q)_L$(L).csv") |> DataFrame)
        Z_data = CSV.File("data/partition_function/partition_function_q$(q)_L$(L).csv") |> DataFrame
        push!(Zs, Z_data[Z_data.beta .== β, :][1, 2])
    end

    p = plot()

    for i in eachindex(qs)
        plot!(data[i][:,1], (data[i][:,2] .- 2 * Zs[i]) / 2 + data[i][:,6], label="\$q = $(qs[i])\$")
    end

    hline!([log(2)/2], label="\$\\ln(2)/2\$")

    xlabel!("\$w\$")
    ylabel!("\$\\ln|\\xi(w)|/N + H(w)\$")

    title!("\$\\beta = $(β)\$")

    display(p)
    save && savefig("figures/weighted/1R2/weighted_action_entropy_1R2_beta$(β)_L$(L).pdf")
end

function plot_1R2_comparison(β, q, L, save=false)

    data = CSV.File("data/weighted/1R2/weighted_1R2_beta$(β)_q$(q)_L$(L).csv") |> DataFrame
    Z_data = CSV.File("data/partition_function/partition_function_q$(q)_L$(L).csv") |> DataFrame
    Z_data = Z_data[Z_data.beta .== β, :][1, :]


    # -I(w)

    ws = LinRange(0, 1, 1000)
    saddle_approx = ws .* 2 .* log(Z_data[4]) + (1 .- ws) .* 2 .* log(Z_data[3]) .- 2 .* Z_data[5]

    p = plot()
    plot!(data[:,1], data[:,2], label="actual")
    plot!(ws, saddle_approx, label="approximation")

    xlabel!("\$w\$")
    ylabel!("\$-I(w)\$")

    title!("1R2 (\$ \\beta = $(β), q = $(q)\$)")

    display(p)
    save && savefig("figures/weighted/1R2/weighted_action_1R2_beta$(β)_L$(L).pdf")
end

function plot_R4_weighted_single_q(β, q, L, plot_2R2 = true, plot_1R4a = true, plot_1R4b = true, save_fig=false)
    inch = 96
    pt = 4/3
    cm = inch / 2.54

    width = 3.25inch
    height = 3 * width / 4
    font = 10pt

    mkpath("figures/weighted/R4/single_q/q$(q)")

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
        f = Figure(size = (width, height), figure_padding = (2, 1, -1, 1), fontsize = font)
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
            label= L"2R2"
        )
        plot_1R4a && lines!(
            ax,
            data_1R4a[:,1],
            data_1R4a[:,2] .- 4 * data_Z[2],
            color = Makie.wong_colors()[2],
            label= L"1R4a"
        )
        plot_1R4b && lines!(
            ax,
            data_1R4b[:,1],
            data_1R4b[:,2] .- 4 * data_Z[2],
            color = Makie.wong_colors()[3],
            label= L"1R4b"
        )

        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0))
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/weighted/R4/single_q/q$(q)/weighted_action_R4_beta$(β)_q$(q)_L$(L).pdf", f)
    end

    # # -I(w) + H(w)

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (2, 1, -1, 1), fontsize = font)
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
            label= L"2R2"
        )
        plot_1R4a && lines!(
            ax,
            data_1R4a[:,1],
            data_1R4a[:,2] .- 4 * data_Z[2] + data_1R4a[:,6],
            color = Makie.wong_colors()[2],
            label= L"1R4a"
        )
        plot_1R4b && lines!(
            ax,
            data_1R4b[:,1],
            data_1R4b[:,2] .- 4 * data_Z[2] + data_1R4b[:,6],
            color = Makie.wong_colors()[3],
            label= L"1R4b"
        )

        axislegend(position = :lb, rowgap = -5, padding = (6, 6, 0, 0))
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/weighted/R4/single_q/q$(q)/weighted_action_entropy_R4_beta$(β)_q$(q)_L$(L).pdf", f)
    end

    # # -p_plus

    with_theme(theme_latexfonts()) do
        f = Figure(size = (width, height), figure_padding = (2, 1, -1, 1), fontsize = font)
        ax = Axis(f[1, 1],
            # title = L"\beta = %$(β), q = %$(q)",
            xlabel = L"$w$",
            ylabel = L"$p_+$",
        )
        plot_2R2 && lines!(
            ax,
            data_2R2[:,1],
            data_2R2[:,5],
            color = Makie.wong_colors()[1],
            label= L"2R2"
        )
        plot_1R4a && lines!(
            ax,
            data_1R4a[:,1],
            data_1R4a[:,5],
            color = Makie.wong_colors()[2],
            label= L"1R4a"
        )
        plot_1R4b && lines!(
            ax,
            data_1R4b[:,1],
            data_1R4b[:,5],
            color = Makie.wong_colors()[3],
            label= L"1R4b"
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
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/weighted/R4/single_q/q$(q)/weighted_p_plus_R4_beta$(β)_q$(q)_L$(L).pdf", f)
    end

    # p = plot()

    # for i in eachindex(qs)
    #     plot!(data[i][:,1], data[i][:,5], label="\$q = $(qs[i])\$")
    # end

    # plot!(LinRange(0, 1, 100), LinRange(0, 1, 100), color=:grey, label="\$w\$")

    # xlabel!("\$w\$")
    # ylabel!("\$p_+\$")

    # title!("1R4a (\$ \\beta = $(β)\$)")

    # display(p)
    # save && savefig("figures/weighted/1R4a/weighted_p_plus_1R4a_beta$(β)_L$(L).pdf")
end


function plot_norm(qs, L, save_fig=false)
    inch = 96
    pt = 4/3
    cm = inch / 2.54

    invphi = (sqrt(5) - 1) / 2

    width = 6.5inch
    height = width * invphi
    font = 12pt

    mkpath("figures/norm_purity/")

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
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/norm_purity/norm_L$(L).pdf", f)
    end
end


function plot_purity(qs, L, save_fig=false)
    inch = 96
    pt = 4/3
    cm = inch / 2.54

    invphi = (sqrt(5) - 1) / 2

    width = 6.5inch
    height = width * invphi
    font = 12pt

    mkpath("figures/norm_purity/")

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
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/norm_purity/purity_L$(L).pdf", f)
    end
end


function plot_norm_vs_purity(qs, L, save_fig=false)
    inch = 96
    pt = 4/3
    cm = inch / 2.54

    invphi = (sqrt(5) - 1) / 2

    width = 6.5inch
    height = width * invphi
    font = 12pt

    mkpath("figures/norm_purity/")

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
        resize_to_layout!(f)
        display(f)
        save_fig && save("figures/norm_purity/norm_purity_L$(L).pdf", f)
    end
end

# plot_1R2_weighted(0.1, [2, 4], 1000, true)
# plot_1R2_weighted(0.2, [2, 4], 1000, true)
# plot_1R2_weighted(0.5, [2, 4, 6, 8], 1000, true)
# plot_1R2_weighted(1.0, [2, 4, 6, 8], 1000, true)
# plot_1R2_weighted(2.0, [2, 4, 6, 8], 1000, true)
# plot_1R2_weighted(5.0, [2, 4, 6, 8], 1000, true)
# plot_1R2_weighted(10.0, [2, 4, 6, 8], 1000, true)
# plot_1R2_weighted(20.0, [2, 4, 6, 8], 1000, true)
# plot_1R2_weighted(50.0, [2, 4, 6, 8], 1000, true)

# plot_1R4a_weighted(0.1, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(0.2, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(0.5, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(1.0, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(2.0, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(5.0, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(10.0, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(20.0, [2, 4, 6, 8], 1000, true)
# plot_1R4a_weighted(50.0, [2, 4, 6, 8], 1000, true)

# plot_R4_weighted_single_q(50.0, 8, 1000, true, true, false, true)

# plot_1_norm(50.0, [2, 4, 6, 8], 1000, false)


# plot_1_norm(20.0, [2, 4, 6, 8], 1000, false)

# plot_1R2_comparison(10., 4, 1000, false)

# plot_R4_weighted(50.0, 4, 1000, false)

# plot_norm([2, 4, 6, 8], 1000, true)
# plot_purity([2, 4, 6, 8], 1000, true)
plot_norm_vs_purity([2, 4], L, true)
