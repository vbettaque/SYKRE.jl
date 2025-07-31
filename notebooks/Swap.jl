using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify, BlockArrays

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier
using SYKRE.SwapMatrix

function rand_time_invariant(L; periodic=false)
    A = zeros(L, L)
    rands = rand(Float64, L-1) / 4 .+ 0.25
    rands = (rands + reverse(rands)) / 2

    for i = 2:L
        for j = 1:(i-1)
            A[i, j] = rands[i - j]
        end
    end

    sign = periodic ? 1 : -1
    A += sign * A'
    A += I / 2
end

function binary_entropy(xs)
    entropy = zeros(length(xs))
    for i = eachindex(xs)
        @assert 0 ≤ xs[i] ≤ 1
        if !iszero(xs[i]) && !isone(xs[i])
            entropy[i] = - xs[i] * log(xs[i]) - (1 - xs[i]) * log(1 - xs[i])
        end
    end
    return entropy
end

function binary_entropy2(xs)
    entropy = zeros(length(xs))
    for i = eachindex(xs)
        @assert 0 ≤ xs[i] ≤ 1
        if !iszero(xs[i]) && !isone(xs[i])
            entropy[i] = - xs[i] * log2(xs[i]) - (1 - xs[i]) * log2(1 - xs[i])
        end
    end
    return entropy
end

function rand_G_init(L, M)
    R = Matrix{Float64}(I, M, M)
    G_minus = rand_time_invariant(L; periodic = false)
    G_init = blockkron(R, G_minus)

    for _ = 2:M
        R = R[[M; 1:(M-1)], :]
        R[1, :] *= -1
        G_plus = rand_time_invariant(L; periodic = true)
        G_init += blockkron(R, G_plus)
    end

    return G_init
end

function generate_swap_data(L, M, q, β, ws)
    @assert iseven(M) && M > 0
    N = 1
    J = 1

    path = "data/swap_matrix/"
    filename = "swap" * string(M ÷ 2) * "_q" * string(q) * "_L" * string(L) * "_beta" * string(β) * ".csv"
    file = path * filename
    !ispath(path) && mkdir(path)
    if !isfile(file)
        touch(file)
        write(file, "weight,log_swap,pf_minus,pf_plus\n")
    end

    syk_Z = SYKData(N, J, q, 1, β)
    G_Z_init = inv(SYKMatrix.differential(L))
    G_Z, Σ_Z = SYKMatrix.schwinger_dyson(G_Z_init, syk_Z; init_lerp = 0.5, lerp_divisor = 2, max_iters=1000)
    logZ = SYKMatrix.log2Z_saddle(G_Z, Σ_Z, syk_Z)

    syk = SYKData(N, J, q, M, β)
    G_init = inv(SYKMatrix.differential(M * L))
    G_init = BlockedArray(G_init, repeat([L], M), repeat([L], M))

    for i in eachindex(ws)
        w = ws[i]
        println(i, " out of ", length(ws), ": w = ", w)

        log_swap = 0
        pf_minus = 0
        pf_plus = 0

        t = isone(i) ? 0.05 : 0.0001

        G, Σ = SwapMatrix.schwinger_dyson(G_init, w, syk; init_lerp = t, lerp_divisor = 10, max_iters=1000)

        p = plot(Gray.(G .- minimum(G)), title="β = $(β), w = $(w)")
        display(p)
        display(G)

        log_swap = SwapMatrix.log2_trace_swap(G, Σ, w, syk) - M * logZ
        pf_minus, pf_plus = SwapMatrix.pfaffians(Σ, syk)

        println("log_swap = ", SwapMatrix.log2_trace_swap(G, Σ, w, syk))

        G_init = G
        # G_init += blockkron([0 -1; 1 0], rand_time_invariant(2 * L; periodic = true))
        # G_init = BlockedArray(G_init, repeat([L], M), repeat([L], M))

        df = DataFrame(weight = w, log_swap = log_swap, pf_minus = pf_minus, pf_plus = pf_plus)
        CSV.write(file, df, append=true)
    end

end

L = 1000
M = 4
q = 4
β = 50
ws = LinRange(0, 1, 21)

generate_swap_data(L, M, q, β, ws)

data = CSV.File("data/swap_matrix/swap2_q4_L1000_beta10.csv") |> DataFrame

entropy = binary_entropy2(ws)
entropy
p = plot(data[:,1], data[:,2], label="q=4, L = 1000, β = 10")
plot!(data[:,1], data[:,2] + entropy, label="saddle + entropy")

xlabel!("\$ w \$")
ylabel!("\$\\log(saddle)\$")

display(p)

G = rand_G_init(5, 2)