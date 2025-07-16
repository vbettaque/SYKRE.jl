using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier

N = 1
J = 1
q = 2
M = 2
L = 2^11
β = 4

syk = SYKData(N, J, q, M, β)

ωs = fftfreq(L, π * L / β)
odd = isodd.(fftfreq(L, L))
even = iseven.(fftfreq(L, L))

I_rep = Matrix{Float64}(I, M, M)

G_init_freq = zeros(ComplexF64, L, 2, 2)

for i = 1:L
    G_init_freq[i, :, :] = [-im * ωs[i] sqrt(ωs[i]^2 + 2 * J^2); -sqrt(ωs[i]^2 + 2 * J^2) -im * ωs[i]] ./ (2 * J^2)
end

det([-im * ωs[2] 0; 0 -im * ωs[2]] - J^2 * G_init_freq[2, :, :])

Σ_freq = J^2 * G_init_freq

dets_odd = [det(-im * ωs[i] * I_rep - Σ_freq[i, :, :]) / det(Σ_freq[1, :, :]) for i=1:L if odd[i]]
det_odd = reduce(*, dets_odd)

dets_even = [det(-im * ωs[i] * I_rep - Σ_freq[i, :, :]) / det(Σ_freq[1, :, :]) for i=1:L if !odd[i]]
det_even = reduce(*, dets_odd)


SREFourier.p_plus(J^2 .* G_init_freq, syk)

fftfreq(8,8)