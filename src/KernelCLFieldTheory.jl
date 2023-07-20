module KernelCLFieldTheory

using CUDA
using Metal
using LinearAlgebra, StatsBase, Statistics
using PreallocationTools
using Parameters
using StochasticDiffEq
using Flux
using Zygote
using Tullio

using Logging: global_logger
using TerminalLoggers: TerminalLogger

using Bootstrap
using Plots
using LaTeXStrings
using Measurements

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
include("LearnKernel.jl")

include("Implementations/imp_ScalarField.jl")
include("Implementations/imp_ScalarField_1D.jl")
include("Implementations/imp_ScalarField_2D.jl")
include("Implementations/imp_ScalarField_3D.jl")

include("PlotScripts.jl")

end # module KernelCLFieldTheory
