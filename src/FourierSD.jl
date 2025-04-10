module FourierSD

using Plots
using FFTW
using ..SYK

function G_SD(Σ::AbstractVector, syk::SYKData)
    L = length(Σ)
    Σ_fft = (2 * syk.β) * ifft(Σ)
    odd = isodd.(fftfreq(L, L))
    ωs = fftfreq(L, π * L / syk.β)
    G_fft = zeros(ComplexF64, L)
    G_fft[odd] = 4 ./ (-im * ωs[odd] - 2 * Σ_fft[odd])
    return real(fft!(G_fft)) / (2 * syk.β)
end

function Σ_SD(G::AbstractVector, syk::SYKData)
	return syk.J^2 * G.^(syk.q - 1)
end

function schwinger_dyson(L, syk::SYKData; Σ_init = zeros(L), max_iters=10000)
	x = 0.5; b = 2; err=0
	Σ = Σ_init
	G = G_SD(Σ, syk)
	for i=1:max_iters
		Σ = Σ_sd(G, q, J)
		G_new = x * G_sd(Σ, β, M, sre=sre) + (1 - x) * G
		err_new = sum(abs.(G_new - G)) / sum(abs.(G))
		isapprox(err_new, 0) && break
		err_new > err && (x /= b)
		err = err_new
		G = G_new
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ, G
end

end
