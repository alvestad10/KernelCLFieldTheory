using KernelCLFieldTheory
using JLD2
using Plots
using Statistics, LinearAlgebra
using LaTeXStrings

Base.ENV["GKSwstype"]="nul"

############################################################################
##  Load and Save the results
############################################################################
DIR = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02/run2"

KP = jldopen(joinpath(DIR,"KP.jld2"))["KP"];


sol_KP = jldopen(joinpath(DIR,"sol_KP_noKernel.jld2"))["sol_KP"]



KP = KernelCLFieldTheory.updateProblem(KP);
############################################################################
##  Calculate the run simulation
############################################################################
scheme = ImplicitEM()#);linsolve = KernelCLFieldTheory.KrylovJL_GMRES())
RS_test = RunSetup(tspan=200.0,
                    tspan_thermalization=10.0, 
                    #saveat=1.0,
                    scheme=scheme, dtmax=1e-4, abstol=1e-4, reltol=1e-4,
                    NTr=10
)


@time sol_KP = run_simulation(KP,RS_test);

# sol_KP_1em5 = sol_KP
# sol_KP_1em3 = sol_KP
# sol_KP_5em4_500 = sol_KP

# sol_KP = sol_KP_1em3

#jldsave(joinpath(DIR,"sol_KP_noKernel_60000.jld2"), sol_KP=sol_KP)

#@time sol_KP = run_simulation(KP,RS_test;T=Float32,gpu=true);
obs_KP = KernelCLFieldTheory.calc_obs(KP,sol_KP)
fig_SK_KP = plotSKContour(KP,sol_KP; obs = obs_KP)
fig_FW_KP = plotFWContour(KP,sol_KP; obs = obs_KP)



jldsave(joinpath(DIR,"obs_KP_noKernel_60000.jld2"), obs_KP=obs_KP)

KernelCLFieldTheory.savefig(fig_SK_KP,joinpath(DIR,"fig_SK_KP_noKernel_60000.pdf"))
KernelCLFieldTheory.savefig(fig_FW_KP,joinpath(DIR,"fig_FW_KP_noKernel_60000.pdf"))




############################################################################
##  Calculate the boundary terms
############################################################################
all_trVs = jldopen(joinpath(DIR,"all_trVs_noKernel.jld2"))["all_trVs"]

Ys = Float64.(collect(0:0.2:20))
BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=collect(0:0.1:8))
@time B1_Omega, B1_Y, trVs = KernelCLFieldTheory.calcBoundaryTerms(sol_KP,BT)#, trVs = trVs)

#trVs_1em5 = trVs 
#trVs_5em4 = trVs
#trVs_5em4_500 = trVs
#trVs_1em3 = trVs
#trVs_5em3 = trVs
#trVs_1em2 = trVs
#trVs_5em2 = trVs
#trVs_1em1 = trVs

trVs = trVs_1em3

begin
    B1 = B1_Omega

    observables = ["xRe", "xIm", "x2Re", "x2Im"] #, "corr0tRe", "corr0tIm"]
    t_steps = KP.model.contour.t_steps

    fig_BT = plot(xlabel=L"\Omega", ylabel=L"B_1(\Omega)";KernelCLFieldTheory.plot_setup(:topright)...)
    for i in eachindex(observables)
        if i > 4 || i < 3
            continue
        end
        _BT_mean = vec(mean(B1[(i-1)*t_steps .+ (1:t_steps),:],dims=1))
        println(_BT_mean)
        scatter!(fig_BT,Ys,_BT_mean, label=observables[i];KernelCLFieldTheory.markers_dict(i)...)
    end
    hline!(fig_BT,[0.0];label=false,KernelCLFieldTheory.solution_line_dict()...)
    fig_BT_Omega = fig_BT
    fig_BT_Omega
end

begin
    B1 = B1_Y

    observables = ["xRe", "xIm", "x2Re", "x2Im", "corr0tRe", "corr0tIm"]
    t_steps = KP.model.contour.t_steps

    fig_BT = plot(xlabel=L"Y", ylabel=L"B_1(Y)";KernelCLFieldTheory.plot_setup(:topright)...)
    for i in eachindex(observables)
        if i > 4 || i < 3
            continue
        end
        _BT_mean = vec(mean(B1[(i-1)*t_steps .+ (1:t_steps),:],dims=1))
        println(_BT_mean)
        scatter!(fig_BT,Ys,_BT_mean, label=observables[i];KernelCLFieldTheory.markers_dict(i)...)
    end
    hline!(fig_BT,[0.0];label=false,KernelCLFieldTheory.solution_line_dict()...)
    fig_BT_Y = fig_BT
    fig_BT_Y
end

jldsave(joinpath(DIR,"BTs_noKernel.jld2"), BT=BT)
jldsave(joinpath(DIR,"B1_Omega_noKernel.jld2"), B1_Omega=B1_Omega)
jldsave(joinpath(DIR,"B1_Y_noKernel.jld2"), B1_Y=B1_Y)
jldsave(joinpath(DIR,"all_trVs_noKernel.jld2"), all_trVs=all_trVs)

KernelCLFieldTheory.savefig(fig_BT_Omega,joinpath(DIR,"fig_BT_Omega_noKernel.pdf"))
KernelCLFieldTheory.savefig(fig_BT_Y,joinpath(DIR,"fig_BT_Y_noKernel.pdf"))














using Measurements

dtmaxes = [1e-1,1e-2,1e-3,5e-4]
BTs = [0.0583 ± 0.0031,0.061 ± 0.003,0.0099 ± 0.0017]
BTs = [0.21 ± 0.0012, 0.065 ± 0.0031,0.004 ± 0.0061,0.0018 ± 0.0068]
BTs = [0.21, 0.065,0.004,0.0018]
fig_BTextrap = scatter(dtmaxes,BTs;yaxis=:log,xaxis=:log,ylim=[1e-5,1],ylabel=L"B_1", xlabel=L"dt^{\max}",label=false, KernelCLFieldTheory.plot_setup()...)

KernelCLFieldTheory.savefig(fig_BTextrap,joinpath(DIR,"fig_BTextrap_2000.pdf"))




dtmaxes = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,1e-5]
BTs = [0.2307 ± 0.0012,0.2179 ± 0.0012,0.1668 ± 0.0024,0.1178 ± 0.0017,0.0058 ± 0.0014,0.002 ± 0.0025,-0.064 ± 0.04, -0.5 ± 0.18]
fig_BTextrap = scatter(dtmaxes,BTs;xaxis=:log, #yaxis=:log,ylim=[1e-4,1],
                        ylabel=L"B_1", xlabel=L"dt^{\max}",label=false, KernelCLFieldTheory.plot_setup()...)
KernelCLFieldTheory.savefig(fig_BTextrap,joinpath(DIR,"fig_BTextrap_noKernel.pdf"))
