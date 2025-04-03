module FourierSD

using Plots
using FFTW


function G_free(β, L)
    ωs = fftshift(fftfreq(L, 2π/Δτ))

end



β = 2
L = 2^10 - 1
Δτ = 2β / L

fftfreq(L, 2π/Δτ)

fftshift(fftfreq(L, 2π/Δτ))

τs = collect((0:L-1) * Δτ)
ωs = fftshift(fftfreq(L, 2π/Δτ))


G = zeros(ComplexF64, L)
G[1:2:L] = 2*im ./ (ωs[1:2:L])
G_ = fft(ifftshift(G)) / 2β

plot(τs, real(G_))


end
