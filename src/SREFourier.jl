module SREFourier

using FFTW
using ..SYK

function det_ratio(Σ_freq::AbstractVector, syk::SYKData)
    L = length(Σ_freq)
    @assert iseven(L)
    K = L ÷ 2
    freqs = fftfreq(L, L)
    odd = isodd.(freqs)
    even = iseven.(freqs)
    ωs = fftfreq(L, π * L / syk.β)
    eig_odd = -im * ωs[odd] - Σ_freq[odd]
    eig_even = -im * ωs[even] - Σ_freq[even]
    eig_ratio = eig_even ./ eig_odd
    ratio = reduce(*, eig_ratio)
    if iseven(K)
        ratio *= conj(-im * ωs[K+1] - Σ_freq[K+1])
    else
        ratio /= conj(-im * ωs[K+1] - Σ_freq[K+1])
    end
    return ratio
end

function p_odd(Σ_freq::AbstractVector, syk::SYKData)
    @assert iseven(syk.M)
    α = syk.M ÷ 2
    ratio = det_ratio(Σ_freq, syk)
    # println("ratio = ", ratio)
    return 1 / (1 + ratio^α)
end

function G_SD_freq(Σ::AbstractVector, syk::SYKData)
    L = length(Σ)
    even = iseven.(fftfreq(L, L))
    odd = isodd.(fftfreq(L, L))
    ωs = fftfreq(L, π * L / syk.β)
    p = p_odd(Σ, syk)
    # println("p = ", p)
    @assert !isnan(p)
    G = zeros(ComplexF64, L)
    if isapprox(p, 1)
        G[odd] = 4 ./ (-im * ωs[odd] - Σ[odd])
        return G
    elseif isapprox(p, 0)
        G[even] = 4 * (1 - p) ./ (-im * ωs[even] - Σ[even])
        return G
    end
    G[even] = 4 * (1 - p) ./ (-im * ωs[even] - Σ[even])
    G[odd] = 4 * p ./ (-im * ωs[odd] - Σ[odd])
    # println("G = ")
    # display(G)
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
    display(Σ_freq)
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
        display(err_new)
		err = err_new
		G_freq = G_freq_new
        G_real = real(FFT * G_freq) / (2 * syk.β)
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ_real, G_real
end

end
