using KernelCLFieldTheory
using JLD2
using Plots
using Statistics, LinearAlgebra
using Random

Random.seed!(123457)


Base.ENV["GKSwstype"]="nul"

############################################################################
##  Load and Save the results
############################################################################
DIR = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02/"

KP = jldopen(joinpath(DIR,"KP_3.jld2"))["KP"];
KP = KernelCLFieldTheory.updateProblem(KP);


scheme = ImplicitEM()
RS_test = RunSetup(tspan=100, 
                    tspan_thermalization=10, 
                    dtmax=5e-3, abstol=5e-3, reltol=5e-3,
                    scheme=scheme
)


@time sol_KP = run_simulation(KP,RS_test; seed=3453423);

jldsave(joinpath(DIR,"sol_KP_3_5000_4.jld2"), sol_KP=sol_KP)

obs_KP = KernelCLFieldTheory.calc_mean_obs(KP,sol_KP)
fig_SK_KP = plotSKContour(KP,sol_KP; obs=obs_KP)
fig_FW_KP = plotFWContour(KP,sol_KP; obs=obs_KP)



jldsave(joinpath(DIR,"obs_KP_5000_3.jld2"), obs_KP=_obs_KP)

KernelCLFieldTheory.savefig(_fig_SK_KP,joinpath(DIR,"fig_SK_KP_5000_3.pdf"))
KernelCLFieldTheory.savefig(_fig_FW_KP,joinpath(DIR,"fig_FW_KP_5000_3.pdf"))
