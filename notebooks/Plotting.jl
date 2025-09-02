using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.SwapMatrix
using .Replicas

purity_2_data = CSV.File("data/sre_purities/purity_M4_q2_L500.csv") |> DataFrame
purity_4_data = CSV.File("data/sre_purities/purity_M4_q4_L500.csv") |> DataFrame
purity_6_data = CSV.File("data/sre_purities/purity_M4_q6_L500.csv") |> DataFrame

p = plot(purity_2_data[:,1], purity_2_data[:,2], label="\$q = 2\$")
plot!(purity_4_data[:,1], purity_4_data[:,2], label="\$q = 4\$")
plot!(purity_6_data[:,1], purity_6_data[:,2], label="\$q = 6\$")

xlabel!("\$β\$")
ylabel!("\$-I_4\$")


sre_2_data = CSV.File("data/sre/sre_M4_q2_L500.csv") |> DataFrame
sre_4_data = CSV.File("data/sre/sre_M4_q4_L500.csv") |> DataFrame
sre_6_data = CSV.File("data/sre/sre_M4_q6_L500.csv") |> DataFrame

p = plot(sre_2_data[:,1], sre_2_data[:,2], label="\$q = 2\$")
plot!(sre_4_data[:,1], sre_4_data[:,2], label="\$q = 4\$")
plot!(sre_6_data[:,1], sre_6_data[:,2], label="\$q = 6\$")

xlabel!("\$β\$")
ylabel!("\$M_2\$")

ylims!(0,2)

weights_2_data = CSV.File("data/sre_weights/weights_M4_q2_L500.csv") |> DataFrame
weights_4_data = CSV.File("data/sre_weights/weights_M4_q4_L500.csv") |> DataFrame
weights_6_data = CSV.File("data/sre_weights/weights_M4_q6_L500.csv") |> DataFrame

p = plot(weights_2_data[:,1], weights_2_data[:,2], label="\$q = 2\$")
plot!(weights_4_data[:,1], weights_4_data[:,2], label="\$q = 4\$")
plot!(weights_6_data[:,1], weights_6_data[:,2], label="\$q = 6\$")

xlabel!("\$β\$")
ylabel!("\$w_{\\mathrm{crit}}\$")

ylims!(0,2)
