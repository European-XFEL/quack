
module Quack

include("solver.jl")
include("gaussian.jl")

export solve!;
export solve_parallel!
export solve_two_pols!;
export solve_parallel_two_pols!;

export solve_gaussian!;
export solve_gaussian_parallel!
#export solve_gaussian_two_pols!;
#export solve_gaussian_parallel_two_pols!;

end
