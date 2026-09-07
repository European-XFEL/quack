# Nonlinear least-squares formulation of the amplitude fit, solved with CurveFit.jl /
# NonlinearSolve.jl (Levenberg-Marquardt by default).
#
# Unknowns x = [Re(E(ω)); Im(E(ω))] on the ω support (length n_x = 2 n_dim).
# Forward model K(x) = y / ‖y‖₂ with y = (e1 x)² + (e2 x)² (calculate_y! in model.jl).
# Residuals:
#   r_obs  = w .* (K(x) - obs)                                      (n_obs rows)
#   r_spec = sqrt(ratio_bins/κ) (|x_j| - (N/S) s_j),  spectrum_mode == 1 (n_dim rows, NaN s_j skipped)
#   r_reg  = sqrt(ratio_bins/κ) x,                    spectrum_mode == 2 (n_x rows)
# which is the objective F(K(x)) + G(x) that the PDHGM iteration in solver.jl targets.
#
# `NLLSSolver` owns the problem buffers and the CurveFit cache; `nlls_solve!` swaps in new data
# and `reinit!`s the cache, so one solver can be reused across Up values and across frames.

import CurveFit
import NonlinearSolve

struct NLLSProblemData
    e1::Matrix{Float64}
    e2::Matrix{Float64}
    obs::Vector{Float64}
    weight::Vector{Float64}
    spectrum::Vector{Float64}
    spectrum_mode::Int
    lambda::Float64          # ratio_bins / kappa
    n_obs::Int
    n_dim::Int
    n_x::Int
    n_res::Int
    # buffers
    z1::Vector{Float64}
    z2::Vector{Float64}
    y::Vector{Float64}
    Jy::Matrix{Float64}
    yJ::Vector{Float64}
end

function nlls_n_res(n_obs::Int, n_x::Int, spectrum_mode::Int)
    if spectrum_mode == 1
        return n_obs + cld(n_x, 2)
    elseif spectrum_mode == 2
        return n_obs + n_x
    end
    return n_obs
end

function NLLSProblemData(n_obs::Int, n_x::Int, spectrum_mode::Int, kappa::Float64)
    n_dim = cld(n_x, 2)
    ratio_bins = n_obs / n_x
    return NLLSProblemData(zeros(n_obs, n_x), zeros(n_obs, n_x), zeros(n_obs), ones(n_obs),
                           fill(NaN, n_dim), spectrum_mode, ratio_bins / kappa,
                           n_obs, n_dim, n_x, nlls_n_res(n_obs, n_x, spectrum_mode),
                           zeros(n_obs), zeros(n_obs), zeros(n_obs),
                           zeros(n_obs, n_x), zeros(n_x))
end

function nlls_set_data!(d::NLLSProblemData, e1::AbstractMatrix, e2::AbstractMatrix,
                        obs::AbstractVector, weight::AbstractVector, spectrum::AbstractVector)
    copyto!(d.e1, e1)
    copyto!(d.e2, e2)
    copyto!(d.obs, obs)
    copyto!(d.weight, weight)
    copyto!(d.spectrum, spectrum)
    return d
end

function nlls_residual!(r::AbstractVector, x::AbstractVector, d::NLLSProblemData)
    ν = calculate_y!(x, d.e1, d.e2, d.z1, d.z2, d.y)
    @inbounds for i in 1:d.n_obs
        r[i] = d.weight[i] * (d.y[i] * ν - d.obs[i])
    end
    sl = sqrt(d.lambda)
    if d.spectrum_mode == 1
        N = calculate_norm_split(d.n_dim, d.spectrum, x)
        S = calculate_norm(d.n_dim, d.spectrum)
        c = N / S
        @inbounds for j in 1:d.n_dim
            if isnan(d.spectrum[j])
                r[d.n_obs + j] = 0.0
            else
                r[d.n_obs + j] = sl * (sqrt(x[j]^2 + x[d.n_dim + j]^2) - c * d.spectrum[j])
            end
        end
    elseif d.spectrum_mode == 2
        @inbounds for j in 1:d.n_x
            r[d.n_obs + j] = sl * x[j]
        end
    end
    return r
end

# J = d r / d x. For the observation block:
#   dK/dx = ν Jy - ν³ y (yᵀ Jy),  Jy = 2 (z1 .* e1 + z2 .* e2)
function nlls_jacobian!(J::AbstractMatrix, x::AbstractVector, d::NLLSProblemData)
    ν = calculate_y!(x, d.e1, d.e2, d.z1, d.z2, d.y)
    @inbounds for j in 1:d.n_x
        @simd for i in 1:d.n_obs
            d.Jy[i, j] = 2.0 * (d.z1[i] * d.e1[i, j] + d.z2[i] * d.e2[i, j])
        end
    end
    mul!(d.yJ, transpose(d.Jy), d.y)
    ν3 = ν^3
    @inbounds for j in 1:d.n_x
        @simd for i in 1:d.n_obs
            J[i, j] = d.weight[i] * (ν * d.Jy[i, j] - ν3 * d.y[i] * d.yJ[j])
        end
    end
    if d.n_res > d.n_obs
        J[d.n_obs + 1:end, :] .= 0.0
    end
    sl = sqrt(d.lambda)
    if d.spectrum_mode == 1
        N = calculate_norm_split(d.n_dim, d.spectrum, x)
        S = calculate_norm(d.n_dim, d.spectrum)
        @inbounds for j in 1:d.n_dim
            if isnan(d.spectrum[j])
                continue
            end
            row = d.n_obs + j
            a = sqrt(x[j]^2 + x[d.n_dim + j]^2)
            if a > 0
                J[row, j] = sl * x[j] / a
                J[row, d.n_dim + j] = sl * x[d.n_dim + j] / a
            end
            # -(s_j/S) dN/dx_k = -(s_j/S) x_k/N over the valid support
            f = sl * d.spectrum[j] / (S * N)
            for k in 1:d.n_dim
                if !isnan(d.spectrum[k])
                    J[row, k] -= f * x[k]
                    J[row, d.n_dim + k] -= f * x[d.n_dim + k]
                end
            end
        end
    elseif d.spectrum_mode == 2
        @inbounds for j in 1:d.n_x
            J[d.n_obs + j, j] = sl
        end
    end
    return J
end

# 1/2 ‖r‖²
function nlls_objective(d::NLLSProblemData, x::AbstractVector)
    r = zeros(d.n_res)
    nlls_residual!(r, x, d)
    return 0.5 * sum(abs2, r)
end

"""
    NLLSSolver(n_obs, n_x, spectrum_mode, kappa; alg, maxiters, kwargs...)

Problem buffers plus an initialised CurveFit cache for one Up slice of a given size.
Reuse it with [`nlls_solve!`](@ref), which swaps in new data and `reinit!`s the cache.
"""
mutable struct NLLSSolver
    d::NLLSProblemData
    cache
end

function NLLSSolver(n_obs::Int, n_x::Int, spectrum_mode::Int, kappa::Float64;
                    alg=NonlinearSolve.LevenbergMarquardt(), maxiters::Int=30, kwargs...)
    d = NLLSProblemData(n_obs, n_x, spectrum_mode, kappa)
    f! = (r, x, p) -> nlls_residual!(r, x, d)
    j! = (J, x, p) -> nlls_jacobian!(J, x, d)
    nf = NonlinearSolve.NonlinearFunction(f!; jac=j!, resid_prototype=zeros(d.n_res))
    prob = CurveFit.NonlinearCurveFitProblem(nf, zeros(n_x), Float64[])
    cache = NonlinearSolve.init(prob, alg; maxiters=maxiters, kwargs...)
    return NLLSSolver(d, cache)
end

"""
    nlls_solve!(s::NLLSSolver, x0, e1, e2, obs, weight, spectrum)

Fit one Up slice starting from `x0`. Returns `(x, objective, sol)`.
"""
function nlls_solve!(s::NLLSSolver, x0::AbstractVector, e1::AbstractMatrix, e2::AbstractMatrix,
                     obs::AbstractVector, weight::AbstractVector, spectrum::AbstractVector)
    nlls_set_data!(s.d, e1, e2, obs, weight, spectrum)
    NonlinearSolve.reinit!(s.cache; u0=Vector{Float64}(x0))
    sol = NonlinearSolve.solve!(s.cache)
    x = Vector{Float64}(sol.u)
    return x, nlls_objective(s.d, x), sol
end

"""
    solve_nlls(x0, e1, e2, obs, weight, spectrum, spectrum_mode; kappa, alg, maxiters, kwargs...)

One-shot fit of a single Up slice (builds a fresh `NLLSSolver`). Returns `(x, objective, sol)`.
"""
function solve_nlls(x0::AbstractVector, e1::AbstractMatrix, e2::AbstractMatrix,
                    obs::AbstractVector, weight::AbstractVector,
                    spectrum::AbstractVector, spectrum_mode::Int;
                    kappa::Float64=200.0, kwargs...)
    s = NLLSSolver(size(e1, 1), size(e1, 2), spectrum_mode, kappa; kwargs...)
    return nlls_solve!(s, x0, e1, e2, obs, weight, spectrum)
end

# Pool of solvers (one per thread) shared by the Up scan and kept across calls, keyed on
# everything that is baked into the cache at `init` time.
struct NLLSPool
    key::Tuple
    solvers::Channel{NLLSSolver}
end

const NLLS_POOL = Ref{Union{Nothing, NLLSPool}}(nothing)
const NLLS_POOL_LOCK = ReentrantLock()

function nlls_pool(n_obs::Int, n_x::Int, spectrum_mode::Int, kappa::Float64; kwargs...)
    key = (n_obs, n_x, spectrum_mode, kappa, kwargs...)
    lock(NLLS_POOL_LOCK) do
        pool = NLLS_POOL[]
        if pool === nothing || pool.key != key
            n = Threads.nthreads()
            ch = Channel{NLLSSolver}(n)
            for _ in 1:n
                put!(ch, NLLSSolver(n_obs, n_x, spectrum_mode, kappa; kwargs...))
            end
            pool = NLLSPool(key, ch)
            NLLS_POOL[] = pool
        end
        return pool
    end
end

"""
    solve_nlls_scan(x0, e1, e2, obs, weight, spectrum, spectrum_mode; kappa, kwargs...)

Fit every Up slice of the 3D `e1`/`e2` arrays (threaded, reusing pooled solvers) and return
`(best_index, x_best, objectives)`.
"""
function solve_nlls_scan(x0::AbstractVector, e1::AbstractArray{Float64,3}, e2::AbstractArray{Float64,3},
                         obs::AbstractVector, weight::AbstractVector,
                         spectrum::AbstractVector, spectrum_mode::Int;
                         kappa::Float64=200.0, kwargs...)
    # copy Python-owned inputs on the main thread: the tasks below must not touch PyArrays
    x0, e1, e2, obs, weight, spectrum = to_array.((x0, e1, e2, obs, weight, spectrum))
    n_Up = size(e1, 3)
    pool = nlls_pool(size(e1, 1), size(e1, 2), spectrum_mode, kappa; kwargs...)
    results = map_over_Up(n_Up) do i
        s = take!(pool.solvers)
        try
            x, obj, _ = nlls_solve!(s, x0, view(e1, :, :, i), view(e2, :, :, i), obs, weight, spectrum)
            (x, obj)
        finally
            put!(pool.solvers, s)
        end
    end
    objs = [r[2] for r in results]
    best = argmin(objs)
    return best, results[best][1], objs
end

"""
    solve_nlls_parallel!(x, e1, e2, obs, weight, spectrum, spectrum_mode; tol, max_iter, kappa)

Entry point mirroring `solve_parallel!`: scans all Up values, writes the best field into `x`
(initial guess on input) and returns the 1-based index of the best Up. `tol` is the relative
step size below which the iteration stops; `max_iter` caps the number of LM steps.
"""
function solve_nlls_parallel!(x::AbstractVector{Float64},
                              e1::AbstractArray{Float64, 3}, e2::AbstractArray{Float64, 3},
                              obs::AbstractVector{Float64},
                              weight::AbstractVector{Float64},
                              spectrum::AbstractVector{Float64},
                              spectrum_mode::Int64;
                              tol::Float64=1e-4, max_iter::Int64=30, kappa::Float64=200.0)::Int64
    best, x_best, _ = solve_nlls_scan(x, e1, e2, obs, weight, spectrum, spectrum_mode;
                                      kappa=kappa, maxiters=max_iter,
                                      termination_condition=NonlinearSolve.RelTerminationMode(),
                                      reltol=tol)
    x[:] .= x_best
    return best
end
