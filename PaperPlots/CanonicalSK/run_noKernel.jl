using KernelCLFieldTheory
using JLD2
using Plots
using Statistics, LinearAlgebra

Base.ENV["GKSwstype"]="nul"

M = ScalarField_from_at(D=1,m=1.0,λ=1.,RT=3.2,β=0.4,at=0.1,n_steps=8,as=0.2,
                        Δβ = 0.5                
                        #ΔE = 0.05              
)

KP_initial = KernelCLFieldTheory.KernelProblem(M);

scheme = ImplicitEM()
RS_test = RunSetup(tspan=3000, 
                    tspan_thermalization=1000, 
                    scheme=scheme
)


@time _sol_noKernel = run_simulation(KP_initial,RS_test);
_fig_SK_noKernel = plotSKContour(KP_initial,_sol_noKernel)
_fig_FW_noKernel = plotFWContour(KP_initial,_sol_noKernel)

_obs_noKernel = KernelCLFieldTheory.calc_obs(KP_initial,_sol_noKernel)


DIR = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02/"

jldsave(joinpath(DIR,"sol_noKernel.jld2"), sol_noKernel=_sol_noKernel)
jldsave(joinpath(DIR,"obs_noKernel.jld2"), obs_noKernel=_obs_noKernel)

KernelCLFieldTheory.savefig(_fig_SK_noKernel,joinpath(DIR,"fig_SK_noKernel.pdf"))
KernelCLFieldTheory.savefig(_fig_FW_noKernel,joinpath(DIR,"fig_FW_noKernel.pdf"))
