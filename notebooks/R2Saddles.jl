using SYKRE
using SYKRE.SYK
using SYKRE.Replicas
using SYKRE.WeightedReplicas
using SYKRE.TFDReplicas

using CairoMakie, DataFrames, CSV, LsqFit, Printf, Latexify, LaTeXStrings, Colors, Tables

function generate_R2_G_plot(β, w, q, L; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    syk = SYKData(1, 1, q, 2, β)
    G_init = Replicas.init(2, L)
    G, _ = WeightedReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
    G_matrix = convert(Matrix{Float64}, G)

    width = 512
    height = 512

    scene = Scene(camera=campixel!, size = (width, height))

    blue = RGB(0,101.0/255,1)
    orange = RGB(1,154.0/255,0)
    grad = cgrad([blue, :gray95, orange], [0.0, 0.5, 1.0])
    image!(scene, 0..width, 0..height, rotr90(G_matrix), colormap = grad)
    return scene, G
end

function generate_1R4_G_plot(β, w, q, L; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    syk = SYKData(1, 1, q, 4, β)
    G_init = Replicas.init(4, L)
    G_init.blocks[:, :, 2] .= 0
    G_init.blocks[:, :, 4] .= 0
    G, _ = WeightedReplicas.schwinger_dyson(G_init, w, syk; init_lerp = init_lerp, lerp_divisor = lerp_divisor, tol = tol, max_iters = max_iters)
    G_matrix = convert(Matrix{Float64}, G)

    width = 512
    height = 512

    scene = Scene(camera=campixel!, size = (width, height))

    blue = RGB(0,101.0/255,1)
    orange = RGB(1,154.0/255,0)
    grad = cgrad([blue, :gray95, orange], [0.0, 0.5, 1.0])
    image!(scene, 0..width, 0..height, rotr90(G_matrix), colormap = grad)
    return scene, G
end

# for β in [5, 20, 50]
#     for w in [0.01, 0.5, 1.0]
#         f, G = generate_R2_G_plot(β, w, q, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
#         if β == 20 && w == 0.5
#             CSV.write("data/G_saddles/R2/G12_beta$(β)_w$(w)_q$(q)_L$(L).csv", Tables.table(G.blocks[:, :, 2]), writeheader=false)
#         end
#         save("figures/G_saddles/R2/G_R2_beta$(β)_w$(w)_q$(q)_L$(L).pdf", f)
#     end
# end

β = 20
w = 0.25
q = 4
L = 2000

f, G = generate_R2_G_plot(β, w, q, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
CSV.write("data/G_saddles/R2/G12_beta$(β)_w$(w)_q$(q)_L$(L).csv", Tables.table(G.blocks[:, :, 2]), writeheader=false)


# β = 2
# w = 0.5
# q = 4
# L = 1000

# f, G = generate_1R4_G_plot(β, w, q, L; init_lerp = 0.01, lerp_divisor = 2, tol=1e-5, max_iters=1000)
# display(f)
