using MKL
using LinearAlgebra
using Base.Threads

using LsqFit

function K(x::AbstractVector{Float64},
           e1::AbstractMatrix{Float64}, e2::AbstractMatrix{Float64})::AbstractVector{Float64}
    "Map input spectrum in x to an observation"
    z1 = e1 * x
    z2 = e2 * x
    y = z1.^2 + z2.^2
    y = y/sqrt(mapreduce(x -> x^2, +, y))
    return y
end

function modelx(omega::AbstractVector{Float64}, m::AbstractVector{Float64})::AbstractVector{ComplexF128}
    "Map Gaussian model to a complex spectrum"
    amplitude_1 = m[1]
    amplitude_2 = m[2]
    precision = m[3]
    omega_1 = m[4]
    omega_2 = m[5]
    phi_1 = m[6]
    phi_2 = m[7]
    return amplitude_1 * exp.(-0.5*precision*(omega .- omega_1).^2 + 1im*(phi_1 .* omega)) + amplitude_2 * exp.(-0.5 * precision * (omega .- omega_2).^2 + 1im*(phi_2 .* omega))
end

function complexify(x::AbstractVector{ComplexF128})::AbstractVector{Float64}
    "Map complex-valued spectrum to complexification of the vector space"
    return cat(real.(x), imag.(x); dims=1)
end

function model_etof(omega::AbstractVector{Float64}, p::AbstractVector{Float64})::AbstractVector{Float64}
    "Obtain prediction in each dimension of the eTOF observation target"
    return K(complexify(modelx(omega, p)))
end

function model_spec(omega::AbstractVector{Float64}, p::AbstractVector{Float64})::AbstractVector{Float64}
    "Obtain prediction in each dimension of the eTOF observation target"
    return = abs.(modelx(omega, p)).^2
end

function K_two_pols!(xA::AbstractVector{Float64}, xS::AbstractVector{Float64},
                     e1A::AbstractMatrix{Float64}, e2A::AbstractMatrix{Float64},
                     e1B::AbstractMatrix{Float64}, e2B::AbstractMatrix{Float64})::AbstractVector{Float64}
    z1A = e1A * xA
    z2A = e2A * xA
    z1B = e1B * (xS - xA)
    z2B = e2B * (xS - xA)
    y = (z1A + z1B).^2 + (z2A + z2B).^2

    y = y/sqrt(mapreduce(x -> x^2, +, y))
    return y
end

function solve_gaussian!(omega::AbstractVector{Float64},
                         y::AbstractVector{Float64}, x::AbstractVector{Float64},
                         e1::AbstractMatrix{Float64}, e2::AbstractMatrix{Float64},
                         obs::AbstractVector{Float64},
                         weight::AbstractVector{Float64},
                         spectrum::AbstractVector{Float64},
                         spectrum_mode::Int64,
                         evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64},
                         evolution_X::AbstractMatrix{Float64};
                         tol::Float64=1e-5, max_iter::Int64=2000,
                         kappa::Float64=5.0)::Float64

    evolution[:] .= 0.0
    evolution_spec[:] .= 0.0
    evolution_X[:,:] .= 0.0
    ratio_bins::Float64 = n_obs/n_x

    # amplitude_1, amplitude_2, precision, omega_1, omega_2, phi_1, phi_2
    p0 = [1.0, 1.0, 1.0, sum(omega)/length(omega), sum(omega)/length(omega), 0.0, 0.0]
    # fit initial condition to spectrum
    # cannot get the phases (to be taken in main fit), so set them to zero initially
    if spectrum_mode >= 0
        g(w, p) = p[1] * exp.(-0.5 * p[3] * (w - p[4]).^2) + p[2] * exp.(-0.5 * p[3] * (w - p[5]).^2)
        fit = curve_fit(g, omega, spectrum, p0)
        p0[1] = abs(fit.param[1])
        p0[2] = abs(fit.param[2])
        p0[3] = fit.param[3]
        p0[4] = fit.param[4]
        p0[5] = fit.param[5]
    end

    if spectrum_mode >= 0
        target = cat(obs, spectrum; dims=1)
        w = cat(ones(obs), ratio_bins/kappa .* ones(spectrum); dims=1)
        model(w, p) = cat(model_etof(w, p), model_spec(w, p); dims=1)
    else
        target = obs
        w = ones(obs)
        model(w, p) = model_etof(w, p)
    end

    fit = curve_fit(model, omega, target, w, p0)
    x = complexify(modelx(omega, fit.param))
    cost_etof = (model_etof(omega, fit.param) - obs).^2
    cost_spec = (model_spec(omega, fit.param) - spectrum).^2

    return cost_etof + ratio_bins/kappa * cost_spec
end

function solve_gaussian_parallel!(omega::AbstractVector{Float64},
                                  y::AbstractVector{Float64}, x::AbstractVector{Float64},
                                  e1::AbstractArray{Float64, 3}, e2::AbstractArray{Float64, 3},
                                  obs::AbstractVector{Float64},
                                  weight::AbstractVector{Float64},
                                  spectrum::AbstractVector{Float64},
                                  spectrum_mode::Int64,
                                  evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64},
                                  evolution_X::AbstractMatrix{Float64};
                                  tol::Float64=1e-5, max_iter::Int64=2000,
                                  kappa::Float64=5.0)::Int64
    n_Up = size(e1)[3]
    omegas = stack(omega for j in 1:n_Up)
    xs = stack(x for j in 1:n_Up)
    ys = stack(y for j in 1:n_Up)
    e = stack(evolution for j in 1:n_Up)
    eS = stack(evolution_spec for j in 1:n_Up)
    eX = stack(evolution_X for j in 1:n_Up)
    e1 = Array(e1)
    e2 = Array(e2)
    obs = Array(obs)
    weight = Array(weight)
    spectrum = Array(spectrum)

    tasks = map(1:n_Up) do i_Up
        Threads.@spawn solve_gaussian!(view(omegas, :, i_Up), view(ys, :, i_Up), view(xs, :, i_Up),
                                       view(e1, :, :, i_Up), view(e2, :, :, i_Up),
                                       view(obs, :), view(weight, :),
                                       view(spectrum, :), spectrum_mode,
                                       view(e, :, i_Up), view(eS, :, i_Up),
                                       view(eX, :, :, i_Up);
                                       tol=tol, max_iter=max_iter, kappa=kappa,
                                       )
    end
    norms = fetch.(tasks)
    best_Up = argmin(norms)
    y[:] .= ys[:, best_Up]
    x[:] .= xs[:, best_Up]
    evolution[:] .= e[:, best_Up]
    evolution_spec[:] .= eS[:, best_Up]
    evolution_X[:,:] .= eX[:, :, best_Up]
    return best_Up
end

