# Pieces shared by the proximal (solver.jl) and least-squares (nlls.jl) solvers:
# forward model, spectrum norms, objective, input conversion and the threaded Up scan.

# y = (e1 x)² + (e2 x)², with z1 = e1 x and z2 = e2 x kept; returns 1/‖y‖₂
function calculate_y!(x::AbstractVector{Float64},
                      e1::AbstractMatrix{Float64}, e2::AbstractMatrix{Float64},
                      z1::AbstractVector{Float64}, z2::AbstractVector{Float64},
                      y::AbstractVector{Float64})::Float64
    mul!(z1, e1, x)
    mul!(z2, e2, x)
    @. y = z1^2 + z2^2

    return 1.0/sqrt(mapreduce(x -> x^2, +, y))
end

# ‖spectrum‖ over the bins that are not NaN
function calculate_norm(n_dim::Int64, spectrum::AbstractVector{Float64})::Float64
    norm::Float64 = 0.0
    @fastmath @inbounds @simd for j in 1:n_dim
        if !(isnan(spectrum[j]))
            norm = norm + spectrum[j]^2
        end
    end
    norm = sqrt(norm)
end

# ‖x‖ of the complexified vector x = [Re; Im] over the bins where the reference spectrum is not NaN
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

# Σ w² (ν y - obs)²: the streaking misfit for a forward model y with normalisation ν
function streaking_error(y::AbstractVector{Float64}, norm_y::Float64,
                         obs::AbstractVector{Float64}, weight::AbstractVector{Float64})::Float64
    err = 0.0
    @inbounds for i in eachindex(y)
        err += weight[i]^2 * (y[i] * norm_y - obs[i])^2
    end
    return err
end

# Inputs arriving from Python are `PyArray`s, which are not `StridedArray`s and so bypass BLAS in
# `mul!`. Copy anything that is not already a Julia `Array`.
to_array(a::Array) = a
to_array(a::AbstractArray{T, N}) where {T, N} = Array{T, N}(a)

# Run f(i_Up) for every Up on Julia threads and return the results in order. BLAS is limited to
# one thread per task so that the per-Up linear algebra does not oversubscribe the cores.
function map_over_Up(f, n_Up::Int)
    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        tasks = map(1:n_Up) do i_Up
            Threads.@spawn f(i_Up)
        end
        return fetch.(tasks)
    finally
        BLAS.set_num_threads(blas_threads)
    end
end
