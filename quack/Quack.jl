
module Quack

using MKL
using LinearAlgebra
using Base.Threads

include("model.jl")
include("solver.jl")
include("nlls.jl")

export solve!;
export solve_parallel!
export solve_nlls_parallel!

end
