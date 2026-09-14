using MKL
using LinearAlgebra
using Base.Threads

#using LsqFit
using Optim
import ForwardDiff
using ADTypes: AutoForwardDiff
import NaNMath

function calculate_spectral_norm(n_dim::Int64, spectrum::AbstractVector{<:Real})::Real
    norm::Float64 = 0.0
    @fastmath @inbounds @simd for j in 1:n_dim
        if !(isnan(spectrum[j]))
            norm = norm + spectrum[j]^2
        end
    end
    norm = sqrt(norm)
end

function K(x::AbstractVector{<:Real},
           e1::AbstractMatrix{<:Real}, e2::AbstractMatrix{<:Real})::AbstractVector{<:Real}
    "Map input spectrum in x to an observation"
    z1 = e1 * x
    z2 = e2 * x
    y = z1.^2 + z2.^2
    y = y/sqrt(mapreduce(x -> x^2, +, y))
    return y
end

function K_two_pols(xA::AbstractVector{<:Real}, xB::AbstractVector{<:Real},
                    e1A::AbstractMatrix{<:Real}, e2A::AbstractMatrix{<:Real},
                    e1B::AbstractMatrix{<:Real}, e2B::AbstractMatrix{<:Real})::AbstractVector{<:Real}
    "Map input spectrum in x to an observation"
    z1A = e1A * xA
    z2A = e2A * xA
    z1B = e1B * xB
    z2B = e2B * xB
    y = (z1A + z1B).^2 + (z2A + z2B).^2
    y = y/sqrt(mapreduce(x -> x^2, +, y))
    return y
end

function modelx(omega::AbstractVector{<:Real}, m::AbstractVector{<:Real})::AbstractVector{<:Real}
    "Map Gaussian model to a complex spectrum"
    x1 = modelx_single(omega, m[1:4])
    x2 = modelx_single(omega, m[5:8])
    return x1 + x2
end

function modelx_single(omega::AbstractVector{<:Real}, m::AbstractVector{<:Real})::AbstractVector{<:Real}
    "Map Gaussian model to a complex spectrum"
    amplitude_1 = exp(m[1])
    precision = exp(m[2])
    omega_1 = m[3]
    phi_1 = m[4]
    spectrum = amplitude_1 * exp.(-0.5*precision*(omega .- omega_1).^2 + 1im*(phi_1 .* omega))
    return cat(real.(spectrum), imag.(spectrum); dims=1)
end

function model_etof(omega::AbstractVector{<:Real}, p::AbstractVector{<:Real},
                    e1::AbstractMatrix{<:Real}, e2::AbstractMatrix{<:Real})::AbstractVector{<:Real}
    "Obtain prediction in each dimension of the eTOF observation target"
    return K(modelx(omega, p), e1, e2)
end

function model_etof_two_pols(omega::AbstractVector{<:Real}, p::AbstractVector{<:Real},
                             e1A::AbstractMatrix{<:Real}, e2A::AbstractMatrix{<:Real},
                             e1B::AbstractMatrix{<:Real}, e2B::AbstractMatrix{<:Real})::AbstractVector{<:Real}
    "Obtain prediction in each dimension of the eTOF observation target"
    xA = modelx_single(omega, p[1:4])
    xB = modelx_single(omega, p[5:8])
    return K_two_pols(xA, xB, e1A, e2A, e1B, e2B)
end

function model_spec(omega::AbstractVector{<:Real}, p::AbstractVector{<:Real})::AbstractVector{<:Real}
    "Obtain prediction in each dimension of the eTOF observation target"
    pred = modelx(omega, p)
    n_s = cld(length(pred), 2)
    pred_spec = sqrt.(pred[1:n_s].^2 + pred[n_s+1:end].^2)
    pred_spec = pred_spec/sqrt(mapreduce(x -> x^2, +, pred_spec))
    return pred_spec
end

function model_spec_two_pols(omega::AbstractVector{<:Real}, p::AbstractVector{<:Real})::AbstractVector{<:Real}
    "Obtain prediction in each dimension of the eTOF observation target"
    xA = modelx_single(omega, p[1:4])
    xB = modelx_single(omega, p[5:8])
    n_s = cld(length(xA), 2)
    pred_spec = sqrt.((xA[1:n_s] + xB[1:n_s]).^2 + (xA[n_s+1:end] + xB[n_s+1:end]).^2)
    pred_spec = pred_spec/sqrt(mapreduce(x -> x^2, +, pred_spec))
    return pred_spec
end

function K_two_pols!(xA::AbstractVector{<:Real}, xS::AbstractVector{<:Real},
                     e1A::AbstractMatrix{<:Real}, e2A::AbstractMatrix{<:Real},
                     e1B::AbstractMatrix{<:Real}, e2B::AbstractMatrix{<:Real})::AbstractVector{<:Real}
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
                         tol::Float64=1e-10, max_iter::Int64=1000,
                         kappa::Float64=200.0)::Float64

    evolution[:] .= 0.0
    evolution_spec[:] .= 0.0
    evolution_X[:,:] .= 0.0
    n_x::Int64 = size(x)[1]
    n_obs::Int64 = size(y)[1]
    n_dim::Int64 = cld(size(x)[1], 2)
    ratio_bins::Float64 = n_obs/n_x

    norm = calculate_spectral_norm(n_dim, spectrum)

    # log(amplitude_1), log(precision_1), omega_1, phi_1, log(amplitude_2), log(precision_2), omega_2, phi_2
    p0 = [0.0, 0.0, sum(omega)/length(omega), 0.0,
          0.0, 0.0, sum(omega)/length(omega), 0.0]
    # fit initial condition to spectrum
    # cannot get the phases (to be taken in main fit), so set them to zero initially
    if spectrum_mode >= 0
        sp0 = cat(p0[1:3], p0[5:7]; dims=1)
        g = p -> sum(exp(p[1]) * exp.(-0.5 * exp(p[2]) * (omega .- p[3]).^2) + exp(p[4]) * exp.(-0.5 * exp(p[5]) * (omega .- p[6]).^2) - spectrum/norm)^2/length(spectrum)
        try
            result = optimize(g, sp0, LBFGS(); autodiff = AutoForwardDiff()) 
            param = Optim.minimizer(result)
            p0[1:3] .= param[1:3]
            p0[5:7] .= param[4:6]
        catch exception
            if isa(exception, SingularException) || isa(exception, ArgumentError)
                println("Singular matrix in initial fit. Starting fit from random initial setting. $(exception)")
            else
                rethrow()
            end
        end
    end

    cost_etof = (model_etof(omega, p0, e1, e2) - obs).^2
    cost_spec = (model_spec(omega, p0) - spectrum/norm).^2

    model = p -> sum((model_etof(omega, p, e1, e2) - obs).^2)/length(obs)
    if spectrum_mode >= 0
        model  = p -> sum((model_etof(omega, p, e1, e2) - obs).^2)/length(obs) + (ratio_bins/kappa)^2 * sum((model_spec(omega, p) - spectrum/norm).^2)/length(spectrum)
    end

    try
        result = optimize(model, p0, LBFGS(); autodiff = AutoForwardDiff()) 
        param = Optim.minimizer(result)
        p0 .= param
        x .= modelx(omega, p0)
    catch exception
        if isa(exception, SingularException) || isa(exception, ArgumentError)
            println("Singular matrix in main fit. Dropping this Up setting. $(exception)")
            x .= NaN
            p0 .= NaN
        else
            rethrow()
        end
    end

    cost_etof = (model_etof(omega, p0, e1, e2) - obs).^2
    cost_spec = (model_spec(omega, p0) - spectrum/norm).^2

    return sum(cost_etof)/length(cost_etof) + (ratio_bins/kappa)^2 * sum(cost_spec)/length(cost_spec)
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
    best_Up = NaNMath.argmin(norms)
    y[:] .= ys[:, best_Up]
    x[:] .= xs[:, best_Up]
    evolution[:] .= e[:, best_Up]
    evolution_spec[:] .= eS[:, best_Up]
    evolution_X[:,:] .= eX[:, :, best_Up]
    return best_Up
end


function solve_gaussian_two_pols!(omega::AbstractVector{Float64},
                                  y::AbstractVector{Float64},
                                  xA::AbstractVector{Float64}, xB::AbstractVector{Float64},
                                  e1A::AbstractMatrix{Float64}, e2A::AbstractMatrix{Float64},
                                  e1B::AbstractMatrix{Float64}, e2B::AbstractMatrix{Float64},
                                  obs::AbstractVector{Float64},
                                  weight::AbstractVector{Float64},
                                  spectrum::AbstractVector{Float64},
                                  spectrum_mode::Int64,
                                  evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64};
                                  tol::Float64=1e-10, max_iter::Int64=1000,
                                  kappa::Float64=200.0)::Float64
        
    evolution[:] .= 0.0
    evolution_spec[:] .= 0.0
    n_x::Int64 = size(xA)[1]
    n_obs::Int64 = size(y)[1]
    n_dim::Int64 = cld(size(xA)[1], 2)
    ratio_bins::Float64 = n_obs/n_x

    norm = calculate_spectral_norm(n_dim, spectrum)

    # log(amplitude_1), log(precision_1), omega_1, phi_1, log(amplitude_2), log(precision_2), omega_2, phi_2
    p0 = [0.0, 0.0, sum(omega)/length(omega), 0.0,
          0.0, 0.0, sum(omega)/length(omega), 0.0]
    # fit initial condition to spectrum
    # cannot get the phases (to be taken in main fit), so set them to zero initially
    if spectrum_mode >= 0
        sp0 = cat(p0[1:3], p0[5:7]; dims=1)
        g = p -> sum(exp(p[1]) * exp.(-0.5 * exp(p[2]) * (omega .- p[3]).^2) + exp(p[4]) * exp.(-0.5 * exp(p[5]) * (omega .- p[6]).^2) - spectrum/norm)^2/length(spectrum)
        try
            result = optimize(g, sp0, LBFGS(); autodiff = AutoForwardDiff()) 
            param = Optim.minimizer(result)
            p0[1:3] .= param[1:3]
            p0[5:7] .= param[4:6]
        catch exception
            if isa(exception, SingularException) || isa(exception, ArgumentError)
                println("Singular matrix in initial fit. Starting fit from random initial setting. $(exception)")
            else
                rethrow()
            end
        end
    end

    cost_etof = (model_etof_two_pols(omega, p0, e1A, e2A, e1B, e2B) - obs).^2
    cost_spec = (model_spec_two_pols(omega, p0) - spectrum/norm).^2

    model = p -> sum((model_etof_two_pols(omega, p, e1A, e2A, e1B, e2B) - obs).^2)/length(obs)
    if spectrum_mode >= 0
        model  = p -> sum((model_etof_two_pols(omega, p, e1A, e2A, e1B, e2B) - obs).^2)/length(obs) + (ratio_bins/kappa)^2 * sum((model_spec_two_pols(omega, p) - spectrum/norm).^2)/length(spectrum)
    end

    try
        result = optimize(model, p0, LBFGS(); autodiff = AutoForwardDiff()) 
        param = Optim.minimizer(result)
        p0 .= param
        xA .= modelx_single(omega, p0[1:4])
        xB .= modelx_single(omega, p0[5:8])
    catch exception
        if isa(exception, SingularException) || isa(exception, ArgumentError)
            println("Singular matrix in main fit. Dropping this Up setting. $(exception)")
            x .= NaN
            p0 .= NaN
        else
            rethrow()
        end
    end

    cost_etof = (model_etof_two_pols(omega, p0, e1A, e2A, e1B, e2B) - obs).^2
    cost_spec = (model_spec_two_pols(omega, p0) - spectrum/norm).^2

    return sum(cost_etof)/length(cost_etof) + (ratio_bins/kappa)^2 * sum(cost_spec)/length(cost_spec)
end

function solve_gaussian_parallel_two_pols!(omega::AbstractVector{Float64},
                                           y::AbstractVector{Float64},
                                           xA::AbstractVector{Float64}, xB::AbstractVector{Float64},
                                           e1A::AbstractArray{Float64, 3}, e2A::AbstractArray{Float64, 3},
                                           e1B::AbstractArray{Float64, 3}, e2B::AbstractArray{Float64, 3},
                                           obs::AbstractVector{Float64},
                                           weight::AbstractVector{Float64},
                                           spectrum::AbstractVector{Float64},
                                           spectrum_mode::Int64,
                                           evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64};
                                           tol::Float64=1e-5, max_iter::Int64=2000,
                                           kappa::Float64=5.0)::Int64
    n_Up = size(e1A)[3]
    omegas = stack(omega for j in 1:n_Up)
    xAs = stack(xA for j in 1:n_Up)
    xBs = stack(xB for j in 1:n_Up)
    ys = stack(y for j in 1:n_Up)
    e = stack(evolution for j in 1:n_Up)
    eS = stack(evolution_spec for j in 1:n_Up)
    e1A = Array(e1A)
    e2A = Array(e2A)
    e1B = Array(e1B)
    e2B = Array(e2B)
    obs = Array(obs)
    weight = Array(weight)
    spectrum = Array(spectrum)

    tasks = map(1:n_Up) do i_Up
        Threads.@spawn solve_gaussian_two_pols!(view(omegas, :, i_Up), view(ys, :, i_Up),
                                                view(xAs, :, i_Up), view(xBs, :, i_Up),
                                                view(e1A, :, :, i_Up), view(e2A, :, :, i_Up),
                                                view(e1B, :, :, i_Up), view(e2B, :, :, i_Up),
                                                view(obs, :), view(weight, :),
                                                view(spectrum, :), spectrum_mode,
                                                view(e, :, i_Up), view(eS, :, i_Up);
                                                tol=tol, max_iter=max_iter, kappa=kappa,
                                                )
    end
    norms = fetch.(tasks)
    best_Up = NaNMath.argmin(norms)
    y[:] .= ys[:, best_Up]
    xA[:] .= xAs[:, best_Up]
    xB[:] .= xBs[:, best_Up]
    evolution[:] .= e[:, best_Up]
    evolution_spec[:] .= eS[:, best_Up]
    return best_Up
end
