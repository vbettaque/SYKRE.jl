module SYKFourier

using LinearAlgebra
using FFTW
using ..SYK

function det_renormed(Σ_freq::AbstractArray, syk::SYKData)
    L, M, M_ = size(Σ_freq)
    @assert M == M_ == syk.M
    @assert isodd(L)

    I_rep = Matrix{Float64}(I, syk.M, syk.M)

    odd = isodd.(fftfreq(L, L))
    ωs = fftfreq(L, π * L / syk.β)

    rep_dets = [det(I_rep - im * Σ_freq[i, :, :] / ωs[i]) for i=1:L if odd[i]]
    det_renorm = reduce(*, rep_dets)

    @assert isapprox(imag(det_renorm), 0; atol=1e-10)
    return 2^syk.M * real(det_renorm)
end

function G_SD_freq(Σ_freq::AbstractArray, syk::SYKData)
    L, M, M_ = size(Σ_freq)
    @assert M == M_ == syk.M
    odd = isodd.(fftfreq(L, L))
    ωs = fftfreq(L, π * L / syk.β)
    G = zeros(ComplexF64, L, M, M)
    I_rep = Matrix{Float64}(I, syk.M, syk.M)
    for i=1:L
        odd[i] || continue
        G[i, :, :] = inv(-im * ωs[i] * I_rep - Σ_freq[i, :, :])
    end
    return G
end

function Σ_SD_real(G_real::AbstractArray, syk::SYKData)
	return syk.J^2 * G_real.^(syk.q - 1)
end

function schwinger_dyson(L, syk::SYKData; Σ_init = zeros(L, syk.M, syk.M), max_iters=100000)
	t = 0.5; b = 2; err=0 # Lerp and lerp refinement factors
	Σ_real = Σ_init
    IFFT = plan_ifft(Σ_real, 1; flags=FFTW.MEASURE)
    Σ_freq = syk.β * IFFT * Σ_real
	G_freq = G_SD_freq(Σ_freq, syk)
    FFT = plan_fft(G_freq, 1; flags=FFTW.MEASURE)
    G_real = real(FFT * G_freq) / syk.β
	for i=1:max_iters
		Σ_real = Σ_SD_real(G_real, syk)
        Σ_freq = syk.β * IFFT * Σ_real
		G_freq_new = t * G_SD_freq(Σ_freq, syk) + (1 - t) * G_freq
		err_new = sum(abs.(G_freq_new - G_freq))
		if isapprox(err_new, 0; atol=1e-10)
            G_real = real(FFT * G_freq_new) / syk.β
            Σ_real = Σ_SD_real(G_real, syk)
            println("Converged after ", i, " iterations")
            break
        end
		if (err_new > err && i > 1)
            t /= b
            i -= 1
            continue
        end
		err = err_new
		G_freq = G_freq_new
        G_real = real(FFT * G_freq) / syk.β
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ_real, G_real
end

function action(Σ_real, G_real, syk::SYKData)
    L, M, M_ = size(Σ_real)
    @assert M == M_ == syk.M
    Δτ = syk.β/L
    Σ_freq = syk.β * ifft(Σ_real, 1)

    prop_term = -log(det_renormed(Σ_freq, syk))/2

    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * syk.β * Δτ * sum(G_real.^syk.q)
    return syk.N * (prop_term + on_shell_term)
end


function logZ(L, syk::SYKData; Σ_init = zeros(L, syk.M, syk.M), max_iters=100000)
    Σ, G = schwinger_dyson(L, syk; Σ_init = Σ_init, max_iters=max_iters)
    return -action(Σ, G, syk), Σ, G
end


function free_energy(L, syk::SYKData; Σ_init = zeros(L, M, M), max_iters=100000)
    logZ_saddle, Σ = logZ(L, syk; Σ_init=Σ_init, max_iters=max_iters)
    return - logZ_saddle / syk.β, Σ
end

end
