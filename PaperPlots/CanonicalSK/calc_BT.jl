using KernelCLFieldTheory
using JLD2
using Plots
using Statistics, LinearAlgebra

Base.ENV["GKSwstype"]="nul"

############################################################################
##  Load and Save the results
############################################################################
DIR = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02/"

KP = jldopen(joinpath(DIR,"KP.jld2"))["KP"]
sol_KP = jldopen(joinpath(DIR,"sol_KP_3.jld2"))["sol_KP"]

scheme = ImplicitEM()
RS_test = RunSetup(tspan=5000, 
                    tspan_thermalization=1000, 
                    scheme=scheme
)


@time _sol_KP = run_simulation(KP,RS_test);
_fig_SK_KP = plotSKContour(KP,_sol_KP)
_fig_FW_KP = plotFWContour(KP,_sol_KP)

_obs_KP = KernelCLFieldTheory.calc_obs(KP,_sol_KP)


jldsave(joinpath(DIR,"sol_KP_5000.jld2"), sol_KP=_sol_KP)
jldsave(joinpath(DIR,"obs_KP_5000.jld2"), obs_KP=_obs_KP)

KernelCLFieldTheory.savefig(_fig_SK_KP,joinpath(DIR,"fig_SK_KP_5000.pdf"))
KernelCLFieldTheory.savefig(_fig_FW_KP,joinpath(DIR,"fig_FW_KP_5000.pdf"))
