module FourierSD

using Plots
using FFTW

β = 1
L = 52 - 1
Δτ = 2β / L
τs = collect((0:L-1) * Δτ)
ωs = fftshift(fftfreq(L, 2π/Δτ))


G = zeros(ComplexF64, L)
G[1:2:L] = 2*im ./ (ωs[1:2:L])
G_ = fft(ifftshift(G))

plot(τs, real(G_))


end
