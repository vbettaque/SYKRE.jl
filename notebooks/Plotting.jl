using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.WeightMatrix
using SYKRE.Replicas


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

plot_1R2_weighted(0.1, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(0.2, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(0.5, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(1.0, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(2.0, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(5.0, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(10.0, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(20.0, [2, 4, 6, 8], 1000, true)
plot_1R2_weighted(50.0, [2, 4, 6, 8], 1000, true)

plot_1R4a_weighted(0.1, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(0.2, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(0.5, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(1.0, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(2.0, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(5.0, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(10.0, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(20.0, [2, 4, 6, 8], 1000, true)
plot_1R4a_weighted(50.0, [2, 4, 6, 8], 1000, true)
