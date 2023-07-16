module KernelCLFieldTheory

using CUDA
using Metal
using LinearAlgebra, StatsBase, Statistics
using PreallocationTools
using Parameters
using StochasticDiffEq

using Logging: global_logger
using TerminalLoggers: TerminalLogger

using Bootstrap


USE_GPU = false
function use_GPU(_use_GPU::Bool)
    global USE_GPU
    USE_GPU = _use_GPU
end
export use_GPU


include("Model.jl")
include("Kernel.jl")
include("Solutions.jl")
include("KernelProblem.jl")
include("RunSetup.jl")

include("implementations/imp_ScalarField_1D.jl")

end # module KernelCLFieldTheory
