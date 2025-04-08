module FourierSD

using Plots
using FFTW

β = 1
J = 1
q = 4
L = 2^15


Δτ = 2β / L
τs = collect((0:L-1) * Δτ)
ωs = fftfreq(L, 2π/Δτ)

odd = isodd.(fftfreq(L, L))
even = iseven.(fftfreq(L, L))

G = zeros(ComplexF64, L)
G[odd] = 4*im ./ ωs[odd]

G_ = fft(G) ./ 2β
Σ_ = J^2 * G_.^(q-1)

Σ = 2β * ifft(Σ_)
G[odd] = 0.5 * G[odd] + 0.5 * 4 ./ (-im * ωs[odd] - 2 * Σ[odd])

G_ = fft(G) ./ 2β
Σ_ = J^2 * G_.^(q-1)

Σ = 2β * ifft(Σ_)
G[odd] = 0.5 * G[odd] + 0.5 * 4 ./ (-im * ωs[odd] - 2 * Σ[odd])

G_ = fft(G) ./ 2β

plot(τs, real(G_))

G = zeros(ComplexF64, L)
G[even] = 4*im ./ ωs[even]
G[1] = 0
G_ = fft(G) ./ 2β
plot(τs, real(G_))


end
