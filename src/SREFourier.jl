module SREFourier

using LinearAlgebra
using FFTW
using ..SYK
using ..SYKFourier

function det_renormed(Σ_freq::AbstractArray, syk::SYKData; odd_modes = true)
    L, M, M_ = size(Σ_freq)
    @assert M == M_ == syk.M
    @assert isodd(L)

    I_rep = Matrix{Float64}(I, syk.M, syk.M)

    odd = isodd.(fftfreq(L, L))
    even = iseven.(fftfreq(L, L))

    ωs = fftfreq(L, π * L / syk.β)


    if odd_modes
        rep_dets = [det(I_rep - im * Σ_freq[i, :, :] / ωs[i]) for i=1:L if odd[i]]
        det_renorm = reduce(*, rep_dets)
        # @assert isapprox(imag(det_renorm), 0; atol=1e-10)
        return 2^syk.M * real(det_renorm)
    end

    iszero(det(Σ_freq[1, :, :])) && return 0

    rep_dets = [det(-im * ωs[i] * I_rep - Σ_freq[i, :, :]) / det(-im * ωs[i] * I_rep - Σ_freq[1, :, :]) for i=1:L if even[i]]
    det_renorm = reduce(*, rep_dets)
    # @assert isapprox(imag(det_renorm), 0; atol=1e-10)
    reg_factor = det(2 * sinh(-syk.β * real(Σ_freq[1, :, :]) / 2))

    return reg_factor * real(det_renorm)
end


function pfaffians(Σ_real, syk::SYKData)
    Σ_freq = syk.β * ifft(Σ_real, 1)
    det_minus = det_renormed(Σ_freq, syk; odd_modes = true)
    det_plus = det_renormed(Σ_freq, syk; odd_modes = false)
    pf_minus = sqrt(det_minus)
    pf_plus = sqrt(det_plus)
    return pf_minus, pf_plus
end

function p_plus(Σ_freq::AbstractArray, syk::SYKData)
    det_minus = det_renormed(Σ_freq, syk; odd_modes = true)
    @assert det_minus >= 0
    pf_minus = sqrt(det_minus)
    det_plus = det_renormed(Σ_freq, syk; odd_modes = false)
    det_plus < 0 && return NaN
    pf_plus = sqrt(det_plus)
    return pf_plus / (pf_minus + pf_plus)
end

function G_SD_freq(Σ_freq::AbstractArray, syk::SYKData)
    L, M, M_ = size(Σ_freq)
    @assert M == M_ == syk.M
    odd = isodd.(fftfreq(L, L))
    ωs = fftfreq(L, π * L / syk.β)
    p = p_plus(Σ_freq, syk)
    display(p)
    # @assert !isnan(p)
    isnan(p) && return NaN
    G = zeros(ComplexF64, L, M, M)
    I_rep = Matrix{Float64}(I, syk.M, syk.M)
    for i=1:L
        if odd[i]
            G[i, :, :] = (1 - p) * inv(-im * ωs[i] * I_rep - Σ_freq[i, :, :])
        else
            G[i, :, :] = p * inv(-im * ωs[i] * I_rep - Σ_freq[i, :, :])
        end
    end
    return G
end

function Σ_SD_real(G_real::AbstractArray, syk::SYKData)
	return syk.J^2 * G_real.^(syk.q - 1)
end

function schwinger_dyson(L, syk::SYKData; Σ_init = zeros(L, syk.M, syk.M), max_iters=100000)
	Σ_real = Σ_init
    IFFT = plan_ifft(Σ_real, 1; flags=FFTW.MEASURE)
    Σ_freq = syk.β * IFFT * Σ_real
	G_freq = G_SD_freq(Σ_freq, syk)
    t = 0.001; b = 2; err=0 # Lerp and lerp refinement factors
    FFT = plan_fft(G_freq, 1; flags=FFTW.MEASURE)
    G_real = real(FFT * G_freq) / syk.β
    Σ_real = Σ_SD_real(G_real, syk)
	for i=1:max_iters
        Σ_freq = syk.β * IFFT * Σ_real
		G_freq_new = t * G_SD_freq(Σ_freq, syk) + (1 - t) * G_freq
		err_new = sum(abs.(G_freq_new - G_freq))
		if isapprox(err_new, 0; atol=1e-9)
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
        G_real_new = real(FFT * G_freq) / syk.β
        Σ_real_new = Σ_SD_real(G_real_new, syk)
        Σ_freq_new = syk.β * IFFT * Σ_real_new
        if any(isnan.(G_SD_freq(Σ_freq_new, syk)))
            t /= b
            i -= 1
            continue
        end
		err = err_new
        println("err = ", err, " t = ", t)
		G_freq = G_freq_new
        G_real = G_real_new
        Σ_freq = Σ_freq_new
        Σ_real = Σ_real_new
        i == max_iters && println("Exceeded iterations!")
	end
	return Σ_real, G_real
end

function action(Σ_real, G_real, syk::SYKData)
    L, M, M_ = size(Σ_real)
    @assert M == M_ == syk.M
    Δτ = syk.β/L
    Σ_freq = syk.β * ifft(Σ_real, 1)

    det_minus = det_renormed(Σ_freq, syk; odd_modes = true)
    det_plus = det_renormed(Σ_freq, syk; odd_modes = false)
    prop_term = -log(sqrt(det_minus) + sqrt(det_plus))

    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * syk.β * Δτ * sum(G_real.^syk.q)
    return syk.N * (prop_term + on_shell_term)
end

function sre(L, syk::SYKData; Σ_M_init = zeros(L, syk.M, syk.M), Σ_2_init = zeros(L, 2, 2), Σ_Z_init = zeros(L, 1, 1), max_iters=100000)
    @assert iseven(syk.M)
    @assert iseven(syk.q)
    @assert isodd(L)
    @assert isodd((L-1) ÷ 2)

    syk_M = syk
    syk_2 = SYKData(syk.N, syk.J, syk.q, 2, syk.β)
    syk_Z = SYKData(syk.N, syk.J, syk.q, 1, syk.β)

    Σ_2, G_2 = schwinger_dyson(L, syk_2; Σ_init=Σ_2_init, max_iters=max_iters)
    Σ_M, G_M = schwinger_dyson(L, syk_M; Σ_init=Σ_M_init, max_iters=max_iters)

    logtr_swap_M = - log(2, ℯ) * action(Σ_M, G_M, syk_M)
    logtr_swap_2 = - log(2, ℯ) * action(Σ_2, G_2, syk_2)
    logZ, Σ_Z = SYKFourier.logZ(L, syk_Z; Σ_init=Σ_Z_init, max_iters=max_iters)

    println("logtr_swap_M = ", logtr_swap_M)
    println("logtr_swap_2 = ", logtr_swap_2)
    println("logZ = ", logZ)
    sre = (logtr_swap_M - logtr_swap_2) / (1 - syk.M/2) + 2 * log(2, ℯ) * logZ
    println("sre = ", sre)
    return sre, Σ_M, Σ_2, Σ_Z
end

end
