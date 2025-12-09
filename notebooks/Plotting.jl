using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.WeightMatrix
using .Replicas


# 1R2

# beta = 0.1

weight_1R2_beta01_q2 = CSV.File("data/paper/weights/1R2/weight_1R2_beta0.1_q2_L1000.csv") |> DataFrame
weight_1R2_beta01_q4 = CSV.File("data/paper/weights/1R2/weight_1R2_beta0.1_q4_L1000.csv") |> DataFrame
weight_1R2_beta01_q6 = CSV.File("data/paper/weights/1R2/weight_1R2_beta0.1_q6_L1000.csv") |> DataFrame
weight_1R2_beta01_q8 = CSV.File("data/paper/weights/1R2/weight_1R2_beta0.1_q8_L1000.csv") |> DataFrame

# -I(w)

p = plot(weight_1R2_beta01_q2[:,1], weight_1R2_beta01_q2[:,2], label="\$q = 2\$")
plot!(weight_1R2_beta01_q4[:,1], weight_1R2_beta01_q4[:,2], label="\$q = 4\$")
plot!(weight_1R2_beta01_q6[:,1], weight_1R2_beta01_q6[:,2], label="\$q = 6\$")
plot!(weight_1R2_beta01_q8[:,1], weight_1R2_beta01_q8[:,2], label="\$q = 8\$")

xlabel!("\$w\$")
ylabel!("\$-I(w)\$")

title!("1R2 (\$ \\beta = 0.1\$)")

# -I(w) + H(w)

p = plot(weight_1R2_beta01_q2[:,1], weight_1R2_beta01_q2[:,2] + weight_1R2_beta01_q2[:,6], label="\$q = 2\$")
plot!(weight_1R2_beta01_q4[:,1], weight_1R2_beta01_q4[:,2] + weight_1R2_beta01_q4[:,6], label="\$q = 4\$")
plot!(weight_1R2_beta01_q6[:,1], weight_1R2_beta01_q6[:,2] + weight_1R2_beta01_q6[:,6], label="\$q = 6\$")
plot!(weight_1R2_beta01_q8[:,1], weight_1R2_beta01_q8[:,2] + weight_1R2_beta01_q8[:,6], label="\$q = 8\$")

xlabel!("\$w\$")
ylabel!("\$-I(w) + H(w)\$")

title!("1R2 (\$ \\beta = 0.1\$)")

# p_+

p = plot(weight_1R2_beta01_q2[:,1], weight_1R2_beta01_q2[:,5], label="\$q = 2\$")
plot!(weight_1R2_beta01_q4[:,1], weight_1R2_beta01_q4[:,5], label="\$q = 4\$")
plot!(weight_1R2_beta01_q6[:,1], weight_1R2_beta01_q6[:,5], label="\$q = 6\$")
plot!(weight_1R2_beta01_q8[:,1], weight_1R2_beta01_q8[:,5], label="\$q = 8\$")
plot!(weight_1R2_beta01_q8[:,1], weight_1R2_beta01_q8[:,1], label="\$w\$")

ylims!(0, 0.4)

xlabel!("\$w\$")
ylabel!("\$p_+(w)\$")

title!("1R2 (\$ \\beta = 0.1\$)")

