using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays


function plot_1R2_weighted(β, qs, L, save=false)

    mkpath("figures/weighted/1R2/")

    data = []
    for q in qs
        push!(data, CSV.File("data/weighted/1R2/weighted_1R2_beta$(β)_q$(q)_L$(L).csv") |> DataFrame)
    end

    # -I(w)

    p = plot()

    for i in eachindex(qs)
        plot!(data[i][:,1], data[i][:,2], label="\$q = $(qs[i])\$")
    end

    xlabel!("\$w\$")
    ylabel!("\$-I(w)\$")

    title!("1R2 (\$ \\beta = $(β)\$)")

    display(p)
    save && savefig("figures/weighted/1R2/weighted_action_1R2_beta$(β)_L$(L).pdf")

    # -I(w) + H(w)

    p = plot()

    for i in eachindex(qs)
        plot!(data[i][:,1], data[i][:,2] + data[i][:,6], label="\$q = $(qs[i])\$")
    end

    xlabel!("\$w\$")
    ylabel!("\$-I(w) + H(w)\$")

    title!("1R2 (\$ \\beta = $(β)\$)")

    display(p)
    save && savefig("figures/weighted/1R2/weighted_action_entropy_1R2_beta$(β)_L$(L).pdf")

    # -p_plus

    p = plot()

    for i in eachindex(qs)
        plot!(data[i][:,1], data[i][:,5], label="\$q = $(qs[i])\$")
    end

    plot!(data[1][:,1], data[1][:,1], color=:grey, label="\$w\$")

    xlabel!("\$w\$")
    ylabel!("\$p_+\$")

    title!("1R2 (\$ \\beta = $(β)\$)")

    display(p)
    save && savefig("figures/weighted/1R2/weighted_p_plus_1R2_beta$(β)_L$(L).pdf")
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

function plot_1R4a_weighted(β, qs, L, save=false)

    mkpath("figures/weighted/1R4a/")

    data = []
    for q in qs
        push!(data, CSV.File("data/weighted/1R4a/weighted_1R4a_beta$(β)_q$(q)_L$(L).csv") |> DataFrame)
    end

    # -I(w)

    p = plot()

    for i in eachindex(qs)
        plot!(data[i][:,1], data[i][:,2], label="\$q = $(qs[i])\$")
    end

    xlabel!("\$w\$")
    ylabel!("\$-I(w)\$")

    title!("1R4a (\$ \\beta = $(β)\$)")

    display(p)
    save && savefig("figures/weighted/1R4a/weighted_action_1R4a_beta$(β)_L$(L).pdf")

    # -I(w) + H(w)

    p = plot()

    for i in eachindex(qs)
        plot!(data[i][:,1], data[i][:,2] + data[i][:,6], label="\$q = $(qs[i])\$")
    end

    xlabel!("\$w\$")
    ylabel!("\$-I(w) + H(w)\$")

    title!("1R4a (\$ \\beta = $(β)\$)")

    display(p)
    save && savefig("figures/weighted/1R4a/weighted_action_entropy_1R4a_beta$(β)_L$(L).pdf")

    # -p_plus

    p = plot()

    for i in eachindex(qs)
        plot!(data[i][:,1], data[i][:,5], label="\$q = $(qs[i])\$")
    end

    plot!(LinRange(0, 1, 100), LinRange(0, 1, 100), color=:grey, label="\$w\$")

    xlabel!("\$w\$")
    ylabel!("\$p_+\$")

    title!("1R4a (\$ \\beta = $(β)\$)")

    display(p)
    save && savefig("figures/weighted/1R4a/weighted_p_plus_1R4a_beta$(β)_L$(L).pdf")
end

# plot_1R2_weighted(0.1, [2, 4, 6, 8], 1000, true)
# plot_1R2_weighted(0.2, [2, 4, 6, 8], 1000, true)
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

# plot_1_norm(50.0, [2, 4, 6, 8], 1000, false)


# plot_1_norm(20.0, [2, 4, 6, 8], 1000, false)

plot_1R2_comparison(10., 4, 1000, false)
