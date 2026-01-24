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
    L, R, R_ = size(Σ_freq)
    @assert R == R_ == syk.M
    odd = isodd.(fftfreq(L, L))
    ωs = fftfreq(L, π * L / syk.β)
    G = zeros(ComplexF64, L, R, R)
    I_rep = Matrix{Float64}(I, syk.R, syk.R)
    for i=1:L
        odd[i] || continue
        G[i, :, :] = inv(-im * ωs[i] * I_rep - Σ_freq[i, :, :])
    end
    return G
end

function Σ_SD_real(G_real::AbstractArray, syk::SYKData)
	return syk.J^2 * G_real.^(syk.q - 1)
end

function schwinger_dyson(G_init_real, syk::SYKData; init_lerp = 0.5, lerp_divisor = 2, tol=1e-5, max_iters=1000)
    L, R, R_ = size(G_init)
    @assert R == R_ == syk.M

    IFFT = plan_ifft(G_real, 1; flags=FFTW.EXHAUSTIVE, timelimit=Inf)

    t = init_lerp

    G_real = G_init_real
    Σ_real = Σ_SD_real(G_real, syk)

    i = 1

    @info "Iteration $(i)" rel_error = err lerp = t

    Σ_freq = syk.β * (IFFT * Σ_real)
    G_new_freq = G_SD_freq(Σ_freq, syk)

    FFT = plan_fft(G_new_freq, 1; flags=FFTW.EXHAUSTIVE, timelimit=Inf)

	G_new_real = real(FFT * G_new_freq) / β
    G_lerp_real = zeros(L, R, R)

    err = sqrt(sum(abs2, G_new_real - G_real)) / sqrt(sum(abs2, G_real))

	while i <= max_iters
        G_lerp_real = t * G_new_real + (1 - t) * G_real

        err_new = sqrt(sum(abs2, G_lerp_real - G_real)) / sqrt(sum(abs2, G_real))

        if err_new < tol
            G_real = G_lerp_real
            Σ_real = Σ_SD(G_real, syk)
            @info "Converged after $(i) iterations"
            break
        end

        if (err_new > err)
            @info "Relative error increased to $(err_new), trying again..."
            t /= lerp_divisor
            continue
        end

        err = err_new

        G_real = G_lerp_real
		Σ_real = Σ_SD_real(G_real, syk)

        i += 1

        if i > max_iters
            @warn "Exceeded iterations!"
            break
        end

        @info "Iteration $(i)" rel_error = err lerp = t

        Σ_freq = syk.β * (IFFT * Σ_real)
		G_new_freq = G_SD_freq(Σ_freq, syk)
        G_new_real = real(FFT * G_new_freq) / β
	end

	return G_real, Σ_real
end

function action(G_real, Σ_real, syk::SYKData)
    L, R, R_ = size(G_real)
    @assert R == R_ == syk.M
    Δτ = syk.β/L

    IFFT = plan_ifft(G_real, 1; flags=FFTW.EXHAUSTIVE, timelimit=Inf)
    Σ_freq = syk.β * (IFFT * Σ_real)

    prop_term = -log(det_renormed(Σ_freq, syk))/2

    on_shell_term = 1/2 * syk.J^2 * (1 - 1/syk.q) * syk.β * Δτ * sum(G_real.^syk.q)
    return syk.N * (prop_term + on_shell_term)
end


log_saddle(G_real, Σ_real, syk::SYKData) = -action(G_real, Σ_real, syk)

log2_saddle(G_real, Σ_real, syk::SYKData) = log(2, ℯ) * log_saddle(G_real, Σ_real, syk)

function free_energy(G_real, Σ_real, syk::SYKData)
    logZ_saddle = log_saddle(G_real, Σ_real, syk)
    return - logZ_saddle / syk.β
end

end
