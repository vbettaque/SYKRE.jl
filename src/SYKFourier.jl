module SYKFourier

using FFTW
using ..SYK

function G_SD_freq(Σ::AbstractVector, syk::SYKData)
    L = length(Σ)
    odd = isodd.(fftfreq(L, L))
    ωs = fftfreq(L, π * L / syk.β)
    G = zeros(ComplexF64, L)
    G[odd] = 4 ./ (-im * ωs[odd] - Σ[odd])
    return G
end

function Σ_SD_real(G::AbstractVector, syk::SYKData)
	return syk.J^2 * G.^(syk.q - 1)
end

function schwinger_dyson(L, syk::SYKData; Σ_init = zeros(L), max_iters=10000)
	t = 0.5; b = 2; err=0 # Lerp and lerp refinement factors
	Σ_real = Σ_init
    IFFT = plan_ifft(Σ_real; flags=FFTW.MEASURE)
    Σ_freq = (2 * syk.β) * IFFT * Σ_real
	G_freq = G_SD_freq(Σ_freq, syk)
    FFT = plan_fft(G_freq, flags=FFTW.MEASURE)
    G_real = real(FFT * G_freq) / (2 * syk.β)
	for i=1:max_iters
		Σ_real = Σ_SD_real(G_real, syk)
        Σ_freq = (2 * syk.β) * IFFT * Σ_real

		G_freq_new = t * G_SD_freq(Σ_freq, syk) + (1 - t) * G_freq
		err_new = sum(abs.(G_freq_new - G_freq)) / sum(abs.(G_freq))
		isapprox(err_new, 0) && break
		err_new > err && (t /= b)
		err = err_new
		G_freq = G_freq_new
        G_real = real(FFT * G_freq) / (2 * syk.β)
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ_real, G_real
end

end
