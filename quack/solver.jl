using MKL
using LinearAlgebra
using Base.Threads

function calculate_y!(x::AbstractVector{Float64},
                      e1::AbstractMatrix{Float64}, e2::AbstractMatrix{Float64},
                      z1::AbstractVector{Float64}, z2::AbstractVector{Float64},
                      y::AbstractVector{Float64})::Float64
    mul!(z1, e1, x)
    mul!(z2, e2, x)
    @. y = z1^2 + z2^2

    return 1.0/sqrt(mapreduce(x -> x^2, +, y))
end

function calculate_y_two_pols!(xA::AbstractVector{Float64}, xS::AbstractVector{Float64},
                               e1A::AbstractMatrix{Float64}, e2A::AbstractMatrix{Float64},
                               e1B::AbstractMatrix{Float64}, e2B::AbstractMatrix{Float64},
                               z1A::AbstractVector{Float64}, z2A::AbstractVector{Float64},
                               z1B::AbstractVector{Float64}, z2B::AbstractVector{Float64},
                               y::AbstractVector{Float64})::Float64
    mul!(z1A, e1A, xA)
    mul!(z2A, e2A, xA)
    mul!(z1B, e1B, (xS - xA))
    mul!(z2B, e2B, (xS - xA))
    @. y = (z1A + z1B)^2 + (z2A + z2B)^2

    return 1.0/sqrt(mapreduce(x -> x^2, +, y))
end

function calculate_norm(n_dim::Int64, spectrum::AbstractVector{Float64})::Float64
    norm::Float64 = 0.0
    @fastmath @inbounds @simd for j in 1:n_dim
        if !(isnan(spectrum[j]))
            norm = norm + spectrum[j]^2
        end
    end
    norm = sqrt(norm)
end

function calculate_norm_split(n_dim::Int64, spectrum_reference::AbstractVector{Float64},
        spectrum_cartesian::AbstractVector{Float64})::Float64
    norm::Float64 = 0.0
    @fastmath @inbounds @simd for j in 1:n_dim
        if !(isnan(spectrum_reference[j]))
            norm = norm + spectrum_cartesian[j]^2 + spectrum_cartesian[n_dim+j]^2
        end
    end
    norm = sqrt(norm)
    return norm
end

function project_indicator!(n_dim::Int64,
                            current_norm::Float64, spectrum_norm::Float64,
                            spectrum_reference::AbstractVector{Float64},
                            x_tmp::AbstractVector{Float64}, x_tmp_proj::AbstractVector{Float64})
   @fastmath @inbounds @simd for j in 1:n_dim
       a::Float64 = atan(x_tmp[n_dim+j], x_tmp[j])
       x_tmp_proj[j] = current_norm/spectrum_norm*spectrum_reference[j]*cos(a)
       x_tmp_proj[n_dim+j] = current_norm/spectrum_norm*spectrum_reference[j]*sin(a)
   end
end

function project_indicator_twopols!(n_dim::Int64,
                            spectrum_norm::Float64,
                            spectrum_reference::AbstractVector{Float64},
                            xA_tmp::AbstractVector{Float64}, xA_tmp_proj::AbstractVector{Float64},
                            xB_tmp::AbstractVector{Float64}, xB_tmp_proj::AbstractVector{Float64},
                             )
    zA = xA_tmp[1:n_dim] + 1im*xA_tmp[n_dim+1:end]
    zB = xB_tmp[1:n_dim] + 1im*xB_tmp[n_dim+1:end]
    zS2 = real.(abs.(zA + zB).^2)
    mask = (zS2 .< 1e-6)
    s2 = (spectrum_reference.^2) ./ (spectrum_norm^2)
    ratio = sqrt.(s2./zS2)
    ratio[mask] .= 1
    @fastmath @inbounds @simd for j in 1:n_dim
        xA_tmp_proj[j] = ratio[j]*xA_tmp[j]
        xB_tmp_proj[j] = ratio[j]*xB_tmp[j]
        xA_tmp_proj[n_dim+j] = ratio[j]*xA_tmp[n_dim+j]
        xB_tmp_proj[n_dim+j] = ratio[j]*xB_tmp[n_dim+j]
    end
    #zA = xA_tmp[1:n_dim] + 1im*xA_tmp[n_dim+1:end]
    #zB = xB_tmp[1:n_dim] + 1im*xB_tmp[n_dim+1:end]
    #s2 = (spectrum_reference.^2) ./ (spectrum_norm^2) .+ 1im*0
    ## s^2 = |alpha zA + zB|^2
    ## alpha^2 |zA|^2 + alpha re(zA zB*) + |zB|^2 - s^2 = 0
    #a = real.(abs.(zA).^2)
    #b = real.(zA .* conj(zB))
    #c = real.(abs.(zB).^2 - s2)
    #delta = max.(b.^2 .- 4*a.*c, 0)
    #alpha_1 = -b./(2*a) + sqrt.(delta)./(2*a)
    #alpha_2 = -b./(2*a) - sqrt.(delta)./(2*a)
    #alpha = alpha_1
    #for k in 1:n_dim
    #    if abs(alpha_1[k] - 1) < abs(alpha_2[k] - 1)
    #        alpha[k] = alpha_1[k]
    #    else
    #        alpha[k] = alpha_2[k]
    #    end
    #end
    #xA_tmp_proj[1:n_dim] .= alpha .* xA_tmp[1:n_dim]
    #xA_tmp_proj[n_dim+1:end] .= alpha .* xA_tmp[n_dim+1:end]
    #xB_tmp_proj .= xB_tmp
end

function project_zeroindicator_twopols!(n_dim::Int64,
                            spectrum_norm::Float64,
                            spectrum_reference::AbstractVector{Float64},
                            xA_tmp::AbstractVector{Float64}, xA_tmp_proj::AbstractVector{Float64},
                            xB_tmp::AbstractVector{Float64}, xB_tmp_proj::AbstractVector{Float64},
                             )
    s2 = (spectrum_reference.^2)
    m = maximum(s2)
    if m > 0
        s2 ./= m
    end
    ratio = similar(s2)
    ratio[s2 .< 0.1] = 0
    ratio[s2 .>= 0.1] = 1
    @fastmath @inbounds @simd for j in 1:n_dim
        xA_tmp_proj[j] = ratio[j]*xA_tmp[j]
        xB_tmp_proj[j] = ratio[j]*xB_tmp[j]
        xA_tmp_proj[n_dim+j] = ratio[j]*xA_tmp[n_dim+j]
        xB_tmp_proj[n_dim+j] = ratio[j]*xB_tmp[n_dim+j]
    end
end

function project_l2norm!(n_dim::Int64, tau::Float64,
                         current_norm::Float64, spectrum_norm::Float64,
                         spectrum_reference::AbstractVector{Float64},
                         x_tmp::AbstractVector{Float64}, x_tmp_proj::AbstractVector{Float64})
    @fastmath @inbounds @simd for j in 1:n_dim
        if !isnan(spectrum_reference[j])
            a::Float64 = atan(x_tmp[n_dim+j], x_tmp[j])
            x_tmp_proj[j] = 1/(1 + tau)*(x_tmp[j] + tau*current_norm/spectrum_norm*spectrum_reference[j]*cos(a))
            x_tmp_proj[n_dim+j] = 1/(1 + tau)*(x_tmp[n_dim+j] + tau*current_norm/spectrum_norm*spectrum_reference[j]*sin(a))
        else
            x_tmp_proj[j] = x_tmp[j]
            x_tmp_proj[n_dim+j] = x_tmp[n_dim+j]
        end
    end
end

function project_zerol2norm_twopols!(n_dim::Int64, tau::Float64,
                         spectrum_norm::Float64,
                         spectrum_reference::AbstractVector{Float64},
                         xA_tmp::AbstractVector{Float64}, xA_tmp_proj::AbstractVector{Float64};
                         threshold::Float64=0.01
    )
    s2 = (spectrum_reference.^2)
    m = maximum(s2)
    if m > 0
        s2 ./= m
    end
    @fastmath @inbounds @simd for j in 1:n_dim
        if !isnan(spectrum_reference[j]) && (s2[j] < threshold)
            xA_tmp_proj[j] = 1/(1 + tau)*xA_tmp[j]
            xA_tmp_proj[n_dim+j] = 1/(1 + tau)*xA_tmp[n_dim+j]
        else
            xA_tmp_proj[j] = xA_tmp[j]
            xA_tmp_proj[n_dim+j] = xA_tmp[n_dim+j]
        end
    end
end

function project_l1!(n_dim::Int64, tau::Float64,
                     x_tmp::AbstractVector{Float64}, x_tmp_proj::AbstractVector{Float64})
    @fastmath @inbounds @simd for j in 1:n_dim
        x_tmp_proj[j] = sign(x_tmp[j])*max(abs(x_tmp[j] - tau), 0)
        x_tmp_proj[n_dim+j] = sign(x_tmp[n_dim+j])*max(abs(x_tmp[n_dim+j] - tau), 0)
    end
end

function project_regularize_l2norm!(n_dim::Int64, tau::Float64,
                                    x_tmp::AbstractVector{Float64}, x_tmp_proj::AbstractVector{Float64})
   @fastmath @inbounds @simd for j in 1:n_dim
       x_tmp_proj[j] = 1/(1 + tau)*x_tmp[j]
       x_tmp_proj[n_dim+j] = 1/(1 + tau)*x_tmp[n_dim+j]
   end
end

function project_l2norm_regularize_edges!(n_dim::Int64, tau::Float64,
                         current_norm::Float64, spectrum_norm::Float64,
                         spectrum_reference::AbstractVector{Float64},
                         x_tmp::AbstractVector{Float64}, x_tmp_proj::AbstractVector{Float64})
    @inbounds @simd for j in 1:n_dim
        if !isnan(spectrum_reference[j])
            a::Float64 = atan(x_tmp[n_dim+j], x_tmp[j])
            x_tmp_proj[j] = 1/(1 + tau)*(x_tmp[j] + tau*current_norm/spectrum_norm*spectrum[j]*cos(a))
            x_tmp_proj[n_dim+j] = 1/(1 + tau)*(x_tmp[n_dim+j] + tau*current_norm/spectrum_norm*spectrum[j]*sin(a))
        else
            x_tmp_proj[j] = 1/(1 + tau)*x_tmp[j]
            x_tmp_proj[n_dim+j] = 1/(1 + tau)*x_tmp[n_dim+j]
        end
    end
end

function solve!(y::AbstractVector{Float64}, x::AbstractVector{Float64},
                e1::AbstractMatrix{Float64}, e2::AbstractMatrix{Float64},
                obs::AbstractVector{Float64},
                weight::AbstractVector{Float64},
                spectrum::AbstractVector{Float64},
                spectrum_mode::Int64,
                evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64},
                evolution_X::AbstractMatrix{Float64};
                tol::Float64=1e-5, max_iter::Int64=2000,
                kappa::Float64=5.0, step::Float64=1.0, alpha::Float64=0.0)::Float64

    sigma::Float64 = 0.1
    tau::Float64 = 0.1
    n_dim::Int64 = cld(size(x)[1], 2)
    n_x::Int64 = size(x)[1]
    n_obs::Int64 = size(y)[1]

    Ds = similar(x)
    Dyv = similar(x)
    s::Float64 = 0
    tmp_obs = similar(y)
    tmp_obs2 = similar(y)
    x_tmp = similar(x)
    x_tmp_proj = similar(x)
    norm_dy::Float64 = 0.0
    current_norm::Float64 = 0.0
    spectrum_norm::Float64 = 0.0
    errs = zeros(Float64, 100)
    errs_scratch = similar(errs)
    z1 = similar(y)
    z2 = similar(y)
    yhat = similar(y)
    norm_yhat::Float64 = 0.0
    evolution[:] .= 0.0
    evolution_spec[:] .= 0.0
    evolution_X[:,:] .= 0.0
    ratio_bins::Float64 = n_obs/n_x

    norm_yhat = calculate_y!(x, e1, e2, z1, z2, yhat)

    spectrum_norm = calculate_norm(n_dim, spectrum)

    for n in 1:max_iter
        # Ds = ∇‖yhat‖² = 2·Dyᵀ·yhat = 4·(e1ᵀ(z1.*yhat) + e2ᵀ(z2.*yhat)).
        # Computed via gemv so the dense Jacobian Dy is never materialized;
        # its elements are recomputed on the fly where needed below.
        mul!(Ds, transpose(e1), z1)
        mul!(Ds, transpose(e2), z2, 1.0, 1.0)

        if mod(n, 25) == 1
            norm_dy = 0.0
            # compute frobenius norm of jacobian dy without storing it.
            # This causes 1 extra computation of dy every 50 iterations
            # but it allows to combine the steps needed to update X in the below computaion
            for j in 1:n_x
                @fastmath @inbounds @simd for i in 1:n_obs
                    norm_dy = norm_dy + (2.0 * (e1[i, j] * z1[i] + e2[i, j] * z2[i]) * norm_yhat - yhat[i] * Ds[j] * norm_yhat^3)^2
                end
            end

            norm_dy = sqrt(norm_dy)*step
            sigma = 0.95/norm_dy
            tau = 0.95/norm_dy
        end

        # X = X - tau * Dyᵀ * (y .* weight)

        # K'(x) = (E1 x)^2 + (E2 x)^2
        # dK_i/dx_j = 2 (E1 x)_i E1_ij + 2 (E2 x)_i E2_ij
        #
        # K(x) = K'(x)/norm, where norm = sqrt(sum(K'(x)))
        # dK_i/dx_j = dK_i/dx_j / norm - 1/2 K'_i/norm^3 sum_k(dK'_k/dx_j)
        # dK_i/dx_j = dK_i/dx1_j / norm - K'_i/norm^3 sum_k (z1_k E1_kj + z2_k E2_kj)
        #
        # x = x - tau (dK/dx)^T y
        #
        # (dK'/dx)^T y = 2 sum_j(y_j z1_j E1_ji + y_j z2_j E2_ji)
        # (dK/dx)^T y = 2 sum_j(y_j z1_j E1_ji + y_j z2_j E2_ji) - sum_k(z1_k E1_ki + z2_k E2_ki) sum_j(K_j y_j) /norm^3
        s = dot(y, yhat)

        @. tmp_obs = y * z1
        mul!(Dyv, transpose(e1), tmp_obs)

        @. tmp_obs = y * z2
        mul!(Dyv, transpose(e2), tmp_obs, 1.0, 1.0)

        @. x_tmp = x - tau * (2.0 * norm_yhat * Dyv - norm_yhat^3 * s * Ds)

        # G = 1/2||X - spectrum||^2
        # proj_{tau G}[Xtmp] = proj_{tau ||.||^2}[Xtmp - spectrum] + spectrum = (1 - tau/||Xtmp - spectrum||^2)_+ (Xtmp - spectrum) +
        # spectrum
        if spectrum_mode == 0
            current_norm = calculate_norm_split(n_dim, spectrum, x_tmp)
            project_indicator!(n_dim, current_norm, spectrum_norm, spectrum, x_tmp, x_tmp_proj)
        elseif spectrum_mode == 1
            current_norm = calculate_norm_split(n_dim, spectrum, x_tmp)
            project_l2norm!(n_dim, (tau/kappa)*ratio_bins, current_norm, spectrum_norm, spectrum, x_tmp, x_tmp_proj)
        elseif spectrum_mode == 2
            project_regularize_l2norm!(n_dim, (tau/kappa)*ratio_bins, x_tmp, x_tmp_proj)
        elseif spectrum_mode == 3
            current_norm = calculate_norm_split(n_dim, spectrum, x_tmp)
            project_l2norm_regularize_edges!(n_dim, (tau/kappa)*ratio_bins, current_norm, spectrum_norm, spectrum, x_tmp, x_tmp_proj)
        else
            @fastmath @inbounds @simd for j in 1:n_dim
                x_tmp_proj[j] = x_tmp[j]
                x_tmp_proj[n_dim+j] = x_tmp[n_dim+j]
            end
        end

        # calculate Xw
        @fastmath @inbounds @simd for j in 1:n_x
            x[j] = 2*x_tmp_proj[j] - x[j]
        end

        norm_yhat = calculate_y!(x, e1, e2, z1, z2, yhat)

        # F(y) = 1/2 ||w y - w obs||^2
        # For a diagonal matrix W with diagonal w:
        # F(y) = 1/2 ||W y - W obs||^2
        # F(y) = 1/2 y^T (W^T W) y - obs^T (W^T W) y + || W obs ||^2
        # For a function H(x) = 1/2 x^T A x + b^T x + c,
        # we have: prox_{l H} (v) = (I + l A)^{-1} (v - l b)
        # Set A = W^T W = diag(w.^2), b = - obs^T (W^T W) = - (diag(w.^2) obs)^T
        # prox_F(v) = (v + sigma w^2 O)/(1+sigma*w^2)
        # update y = prox_F*(v)
        # v = y + sigma*K
        @. y = y + sigma*(yhat * norm_yhat)
        # prox_F* = v - prox_F(v)
        # prox_F* = (v + sigma w^2 v - v - sigma w^2 O)/(1 + sigma w^2)
        # prox_F* = (sigma w^2 v - sigma w^2 O)/(1 + sigma w^2)
        # prox_F* = sigma w^2/(1 + sigma w^2) (v - O)
        @. y = sigma * weight^2/(1 + sigma * weight^2) * (y - obs)

        # compute errs and terminate loop if error limit is reached
        circshift!(errs_scratch, errs, -1)
        errs, errs_scratch = errs_scratch, errs
        errs[1] = 0.5 * mapreduce(x -> x^2, +, y)

        # store streaking error
        # Note that we use an explicit loop rather than a mapreduce closure to
        # prevent closure-boxing.
        streak_err = 0.0
        for i in 1:n_obs
            streak_err += weight[i]^2 * (yhat[i] * norm_yhat - obs[i])^2
        end
        evolution[n] = streak_err / n_obs

        # store spectrum
        evolution_X[n,:] = x

        # store spectrum error
        current_norm = calculate_norm_split(n_dim, spectrum, x)
        evolution_spec[n] = 0.0
        @fastmath @inbounds @simd for j in 1:n_dim
            if !isnan(spectrum[j])
                evolution_spec[n] = evolution_spec[n] + (sqrt(x[j]^2 + x[n_dim+j]^2)/current_norm - spectrum[j]/spectrum_norm)^2
            end
        end
        evolution_spec[n] = evolution_spec[n]/n_dim

        if n > 500
            err_diff = abs(errs[100] - errs[1])
            if (isnan(err_diff) || isinf(err_diff))
                completed_iter = n
                evolution[n:end] .= evolution[n]
                evolution_spec[n:end] .= evolution_spec[n]
                break
            end
            if err_diff < tol
                completed_iter = n
                evolution[n:end] .= evolution[n]
                evolution_spec[n:end] .= evolution_spec[n]
                break
            end
        end
    end

    #norm_yhat = calculate_y!(x, e1, e2, z1, z2, yhat)
    #return sum((yhat.*norm_yhat - obs/sqrt(sum(obs.^2))).^2)
    return evolution[end] + ratio_bins/kappa * evolution_spec[end]
end

function solve_parallel!(y::AbstractVector{Float64}, x::AbstractVector{Float64},
                e1::AbstractArray{Float64, 3}, e2::AbstractArray{Float64, 3},
                obs::AbstractVector{Float64},
                weight::AbstractVector{Float64},
                spectrum::AbstractVector{Float64},
                spectrum_mode::Int64,
                evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64},
                evolution_X::AbstractMatrix{Float64};
                tol::Float64=1e-5, max_iter::Int64=2000,
                kappa::Float64=5.0, step::Float64=1.0, alpha::Float64=0.0)::Int64
    n_Up = size(e1)[3]
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
        Threads.@spawn solve!(view(ys, :, i_Up), view(xs, :, i_Up),
                              view(e1, :, :, i_Up), view(e2, :, :, i_Up),
                              view(obs, :), view(weight, :),
                              view(spectrum, :), spectrum_mode,
                              view(e, :, i_Up), view(eS, :, i_Up),
                              view(eX, :, :, i_Up);
                              tol=tol, max_iter=max_iter, kappa=kappa,
                              step=step, alpha=alpha)
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

function solve_two_pols!(y::AbstractVector{Float64},
                         xA::AbstractVector{Float64}, xB::AbstractVector{Float64},
                         e1A::AbstractMatrix{Float64}, e2A::AbstractMatrix{Float64},
                         e1B::AbstractMatrix{Float64}, e2B::AbstractMatrix{Float64},
                         obs::AbstractVector{Float64},
                         weight::AbstractVector{Float64},
                         spectrum::AbstractVector{Float64},
                         spectrum_mode::Int64,
                         evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64},
                         ; #evolution_X::AbstractMatrix{Float64};
                         tol::Float64=1e-5, max_iter::Int64=2000,
                         kappa::Float64=5.0, step::Float64=1.0, alpha::Float64=1e-5)::Float64

    sigma::Float64 = 0.1
    tau::Float64 = 0.1
    n_dim::Int64 = cld(size(xA)[1], 2)
    n_x::Int64 = size(xA)[1]
    n_obs::Int64 = size(y)[1]

    DsA = similar(xA)
    DsS = similar(xB)
    DyvA = similar(xA)
    DyvS = similar(xB)
    s::Float64 = 0
    tmp_obs = similar(y)
    tmp_obs2 = similar(y)
    xA_tmp = similar(xA)
    xA_tmp_proj = similar(xA)
    xS_tmp = similar(xB)
    xS_tmp_proj = similar(xB)
    xS = copy(xB)
    xS .= xS + xA

    norm_dy::Float64 = 0.0
    current_norm::Float64 = 0.0
    spectrum_norm::Float64 = 0.0
    errs = zeros(Float64, 100)
    errs_scratch = similar(errs)
    z1A = similar(y)
    z2A = similar(y)
    z1B = similar(y)
    z2B = similar(y)
    yhat = similar(y)
    norm_yhat::Float64 = 0.0
    evolution[:] .= 0.0
    evolution_spec[:] .= 0.0
    #evolution_X[:,:] .= 0.0
    ratio_bins::Float64 = n_obs/n_x

    norm_yhat = calculate_y_two_pols!(xA, xS,
                                      e1A, e2A,
                                      e1B, e2B,
                                      z1A, z2A,
                                      z1B, z2B,
                                      yhat)

    spectrum_norm = calculate_norm(n_dim, spectrum)

    for n in 1:max_iter
        # Ds = ∇‖yhat‖² = 2·Dyᵀ·yhat = 4·(e1ᵀ(z1.*yhat) + e2ᵀ(z2.*yhat)).
        # Computed via gemv so the dense Jacobian Dy is never materialized;
        # its elements are recomputed on the fly where needed below.
        mul!(DsA, transpose(e1A - e1B), z1A + z1B)
        mul!(DsA, transpose(e2A - e2B), z2A + z2B, 1.0, 1.0)
        mul!(DsS, transpose(e1B), z1A + z1B)
        mul!(DsS, transpose(e2B), z2A + z2B, 1.0, 1.0)

        if mod(n, 25) == 1
            norm_dy = 0.0
            # compute frobenius norm of jacobian dy without storing it.
            # This causes 1 extra computation of dy every 50 iterations
            # but it allows to combine the steps needed to update X in the below computaion
            for j in 1:n_x
                @fastmath @inbounds @simd for i in 1:n_obs
                    #norm_dy = norm_dy + (2.0 * ((e1A[i, j] - e1B[i, j]) * (z1A[i] + z1B[i])
                    #                          + (e2A[i, j] - e2B[i, j]) * (z2A[i] + z2B[i])
                    #                          + e1B[i, j] * (z1A[i] + z1B[i])
                    #                          + e2B[i, j] * (z2A[i] + z2B[i])) * norm_yhat
                    #                     - yhat[i] * DsA[j] * norm_yhat^3
                    #                     - yhat[i] * DsS[j] * norm_yhat^3)^2
                    # dK/dxA
                    norm_dy = norm_dy + (2.0 * ((e1A[i, j] - e1B[i, j]) * (z1A[i] + z1B[i]) + (e2A[i, j] - e2B[i, j]) * (z2A[i] + z2B[i])) * norm_yhat - yhat[i] * DsA[j] * norm_yhat^3)^2
                    # dK/dxS
                    norm_dy = norm_dy + (2.0 * (e1B[i, j] * (z1A[i] + z1B[i]) + e2B[i, j] * (z2A[i] + z2B[i])) * norm_yhat - yhat[i] * DsS[j] * norm_yhat^3)^2
                end
            end

            norm_dy = sqrt(norm_dy)*step
            sigma = 0.95/norm_dy
            tau = 0.95/norm_dy
        end

        # X = X - tau * Dyᵀ * (y .* weight)

        # K'(x) = (E1a xa + E1b xb)^2 + (E2a xa + E2b xb)^2
        # K'(x) = (E1a xa + E1b (xs - xa))^2 + (E2a xa + E2b (xs - xa))^2
        # dK'_i/dxa_j = 2 (E1a xa + E1b xb)_i (E1a - E1b)_ij + 2 (E2a xa + E2b xb)_i (E2a - E2b)_ij
        # dK_i/dxs_j = 2 (E1a xa + E1b xb)_i E1b_ij + 2 (E2a xa + E2b xb)_i E2b
 
        # K(x) = K'(x)/norm, where norm = sqrt(sum(K'(x)))
        # dK_i/dx_j = dK'_i/dx_j / norm - 1/2 K'_i/norm^3 sum_k(dK'_k/dx_j)
        # dK_i/dxa_j = dK'_i/dxa_j / norm - K'_i/norm^3 sum_k ((z1a + z1b)_k (E1a - E1b)_kj + (z2a + z2b)_k (E2a - E2b)_kj)
        # dK_i/dxs_j = dK'_i/dxs_j / norm - K'_i/norm^3 sum_k ((z1a + z1b)_k (E1b_kj) + (z2a + z2b)_k (E2b_kj))

        # x = x - tau (dK/dx)^T y
        # (dK'/dxA)^T y = 2 sum_j(y_j (z1a + z1b)_j (E1a - E1b)_ji + y_j (z2a + z2b)_j (E2a - E2b)_ji)
        # (dK/dxA)^T y = 2 sum_j(y_j (z1a + z1b)_j (E1a - E1b)_ji + y_j (z2a + z2b)_j (E2a - E2b)_ji) - sum_k((z1a + z1b)_k (E1a - E1b)_ki + (z2b + z2b)_k (E2a - E2b)_ki) sum_j(K_j y_j) /norm^3
        # (dK/dxS)^T y = 2 sum_j(y_j (z1a + z1b)_j E1b_ji + y_j (z2a + z2b)_j E2b_ji) - sum_k((z1a + z1b)_k E1b_ki + (z2a + z2b)_k E2b_kj) sum_j(K_j y_j) /norm^3
 
        s = dot(y, yhat)

        @. tmp_obs = y * (z1A + z1B)
        mul!(DyvA, transpose(e1A - e1B), tmp_obs)

        @. tmp_obs = y * (z2A + z2B)
        mul!(DyvA, transpose(e2A - e2B), tmp_obs, 1.0, 1.0)

        @. xA_tmp = xA - tau * (2.0 * norm_yhat * DyvA - norm_yhat^3 * s * DsA)

        @. tmp_obs = y * (z1A + z1B)
        mul!(DyvS, transpose(e1B), tmp_obs)

        @. tmp_obs = y * (z2A + z2B)
        mul!(DyvS, transpose(e2B), tmp_obs, 1.0, 1.0)

        @. xS_tmp = xS - tau * (2.0 * norm_yhat * DyvS - norm_yhat^3 * s * DsS)

        # restrict on the sum spectrum
        # G = 1/2||X_s - spectrum||^2
        # proj_{tau G}[Xtmp] = proj_{tau ||.||^2}[Xtmp - spectrum] + spectrum = (1 - tau/||Xtmp - spectrum||^2)_+ (Xtmp - spectrum) +
        if spectrum_mode == 0
            current_norm = calculate_norm_split(n_dim, spectrum, xS_tmp)
            project_indicator!(n_dim, current_norm, spectrum_norm, spectrum, xS_tmp, xS_tmp_proj)
            #project_zerol2norm_twopols!(n_dim, (tau/kappa)*ratio_bins,
            #                            spectrum_norm,
            #                            spectrum,
            #                            xA_tmp, xA_tmp_proj)
            #project_regularize_l2norm!(n_dim, (tau/kappa)*ratio_bins/100,
            #                           xA_tmp, xA_tmp_proj)
        elseif spectrum_mode == 1
            current_norm = calculate_norm_split(n_dim, spectrum, xS_tmp)
            project_l2norm!(n_dim, (tau/kappa)*ratio_bins, current_norm, spectrum_norm, spectrum, xS_tmp, xS_tmp_proj)
            #project_zerol2norm_twopols!(n_dim, (tau/kappa)*ratio_bins,
            #                            spectrum_norm,
            #                            spectrum,
            #                            xA_tmp, xA_tmp_proj)
            #project_regularize_l2norm!(n_dim, (tau/kappa)*ratio_bins/100,
            #                           xA_tmp, xA_tmp_proj)
        elseif spectrum_mode == 2
            project_regularize_l2norm!(n_dim, (tau/kappa)*ratio_bins, xS_tmp, xS_tmp_proj)
            #project_zerol2norm_twopols!(n_dim, (tau/kappa)*ratio_bins,
            #                            spectrum_norm,
            #                            spectrum,
            #                            xA_tmp, xA_tmp_proj)
            #project_regularize_l2norm!(n_dim, (tau/kappa)*ratio_bins/100,
            #                           xA_tmp, xA_tmp_proj)
        elseif spectrum_mode == 3
            current_norm = calculate_norm_split(n_dim, spectrum, x_tmp)
            project_l2norm_regularize_edges!(n_dim, (tau/kappa)*ratio_bins, current_norm, spectrum_norm, spectrum, xS_tmp, xS_tmp_proj)
            #project_zerol2norm_twopols!(n_dim, (tau/kappa)*ratio_bins,
            #                            spectrum_norm,
            #                            spectrum,
            #                            xA_tmp, xA_tmp_proj)
            #project_regularize_l2norm!(n_dim, (tau/kappa)*ratio_bins/100,
            #                           xA_tmp, xA_tmp_proj)
        else
            @fastmath @inbounds @simd for j in 1:n_dim
                xS_tmp_proj[j] = xS_tmp[j]
                xS_tmp_proj[n_dim+j] = xS_tmp[n_dim+j]
            end
        end
        @fastmath @inbounds @simd for j in 1:n_dim
            xA_tmp_proj[j] = xA_tmp[j]
            xA_tmp_proj[n_dim+j] = xA_tmp[n_dim+j]
        end

        # calculate Xw
        @fastmath @inbounds @simd for j in 1:n_x
            xA[j] = 2*xA_tmp_proj[j] - xA[j]
            xS[j] = 2*xS_tmp_proj[j] - xS[j]
        end

        norm_yhat = calculate_y_two_pols!(xA, xS,
                                          e1A, e2A,
                                          e1B, e2B,
                                          z1A, z2A,
                                          z1B, z2B,
                                          yhat)

        # F(y) = 1/2 ||w y - w obs||^2
        # For a diagonal matrix W with diagonal w:
        # F(y) = 1/2 ||W y - W obs||^2
        # F(y) = 1/2 y^T (W^T W) y - obs^T (W^T W) y + || W obs ||^2
        # For a function H(x) = 1/2 x^T A x + b^T x + c,
        # we have: prox_{l H} (v) = (I + l A)^{-1} (v - l b)
        # Set A = W^T W = diag(w.^2), b = - obs^T (W^T W) = - (diag(w.^2) obs)^T
        # prox_F(v) = (v + sigma w^2 O)/(1+sigma*w^2)
        # update y = prox_F*(v)
        # v = y + sigma*K
        @. y = y + sigma*(yhat * norm_yhat)
        # prox_F* = v - prox_F(v)
        # prox_F* = (v + sigma w^2 v - v - sigma w^2 O)/(1 + sigma w^2)
        # prox_F* = (sigma w^2 v - sigma w^2 O)/(1 + sigma w^2)
        # prox_F* = sigma w^2/(1 + sigma w^2) (v - O)
        @. y = sigma * weight^2/(1 + sigma * weight^2) * (y - obs)

        # compute errs and terminate loop if error limit is reached
        circshift!(errs_scratch, errs, -1)
        errs, errs_scratch = errs_scratch, errs
        errs[1] = 0.5 * mapreduce(x -> x^2, +, y)

        # store streaking error
        # Note that we use an explicit loop rather than a mapreduce closure to
        # prevent closure-boxing.
        streak_err = 0.0
        for i in 1:n_obs
            streak_err += weight[i]^2 * (yhat[i] * norm_yhat - obs[i])^2
        end
        evolution[n] = streak_err / n_obs

        # store spectrum
        #evolution_X[n,:] = x

        # store spectrum error
        current_norm = calculate_norm_split(n_dim, spectrum, xS_tmp)
        evolution_spec[n] = 0.0
        @inbounds @simd for j in 1:n_dim
            if !isnan(spectrum[j])
                evolution_spec[n] = evolution_spec[n] + (sqrt(xS[j]^2 + xS[n_dim+j]^2)/current_norm - spectrum[j]/spectrum_norm)^2
            end
        end
        evolution_spec[n] = evolution_spec[n]/n_dim

        #println("n=$n, rmse=$(evolution[n]), spec=$(evolution_spec[n])")

        if n > 500
            err_diff = abs(errs[100] - errs[1])
            if (isnan(err_diff) || isinf(err_diff))
                completed_iter = n
                evolution[n:end] .= evolution[n]
                evolution_spec[n:end] .= evolution_spec[n]
                break
            end
            if err_diff < tol
                completed_iter = n
                evolution[n:end] .= evolution[n]
                evolution_spec[n:end] .= evolution_spec[n]
                break
            end
        end
    end
    # calculate xB
    xB .= xS - xA

    return evolution[end] #+ ratio_bins/kappa * evolution_spec[end]
end

function solve_parallel_two_pols!(y::AbstractVector{Float64},
                                  xA::AbstractVector{Float64}, xB::AbstractVector{Float64},
                                  e1A::AbstractArray{Float64, 3}, e2A::AbstractArray{Float64, 3},
                                  e1B::AbstractArray{Float64, 3}, e2B::AbstractArray{Float64, 3},
                                  obs::AbstractVector{Float64},
                                  weight::AbstractVector{Float64},
                                  spectrum::AbstractVector{Float64},
                                  spectrum_mode::Int64,
                                  evolution::AbstractVector{Float64}, evolution_spec::AbstractVector{Float64},
                                  ;#evolution_X::AbstractMatrix{Float64};
                                  tol::Float64=1e-5, max_iter::Int64=2000,
                                  kappa::Float64=5.0, step::Float64=1.0, alpha::Float64=0.0)::Int64
    n_Up = size(e1A)[3]
    xAs = stack(xA for j in 1:n_Up)
    xBs = stack(xB for j in 1:n_Up)
    ys = stack(y for j in 1:n_Up)
    e = stack(evolution for j in 1:n_Up)
    eS = stack(evolution_spec for j in 1:n_Up)
    #eX = stack(evolution_X for j in 1:n_Up)
    e1A = Array(e1A)
    e2A = Array(e2A)
    e1B = Array(e1B)
    e2B = Array(e2B)
    obs = Array(obs)
    weight = Array(weight)
    spectrum = Array(spectrum)

    tasks = map(1:n_Up) do i_Up
        Threads.@spawn solve_two_pols!(view(ys, :, i_Up),
                                       view(xAs, :, i_Up), view(xBs, :, i_Up),
                                       view(e1A, :, :, i_Up), view(e2A, :, :, i_Up),
                                       view(e1B, :, :, i_Up), view(e2B, :, :, i_Up),
                                       view(obs, :), view(weight, :),
                                       view(spectrum, :), spectrum_mode,
                                       view(e, :, i_Up), view(eS, :, i_Up),
                                       ;#view(eX, :, :, i_Up);
                                       tol=tol, max_iter=max_iter, kappa=kappa,
                                       step=step, alpha=alpha)
    end
    norms = fetch.(tasks)
    best_Up = argmin(norms)
    y[:] .= ys[:, best_Up]
    xA[:] .= xAs[:, best_Up]
    xB[:] .= xBs[:, best_Up]
    evolution[:] .= e[:, best_Up]
    evolution_spec[:] .= eS[:, best_Up]
    #evolution_X[:,:] .= eX[:, :, best_Up]
    return best_Up
end

