using CSV, DataFrames, Statistics, Plots, LinearAlgebra, FFTW, Latexify

using SYKRE
using SYKRE.SYK
using SYKRE.SYKMatrix
using SYKRE.SREMatrix
using SYKRE.SYKFourier
using SYKRE.SREFourier

N = 1
J = 2
q = 2
M = 2
L_freq = 2^11-1
L_matrix = 2000
β = 4

syk = SYKData(N, J, q, M, β)

ωs = fftfreq(L_freq, π * L_freq / β)
odd = isodd.(fftfreq(L_freq, L_freq))

G_init_freq = zeros(ComplexF64, L_freq, 2, 2)

for i = 1:L_freq
    G_init_freq[i, :, :] = [-im * ωs[i] sqrt(ωs[i]^2 + 2 * J^2); -sqrt(ωs[i]^2 + 2 * J^2) -im * ωs[i]] ./ (2 * J^2)
end

det([-im * ωs[2] 0; 0 -im * ωs[2]] - J^2 * G_init_freq[2, :, :])

SREFourier.p_plus(J^2 .* G_init_freq, syk)
