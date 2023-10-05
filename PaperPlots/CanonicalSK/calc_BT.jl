using KernelCLFieldTheory
using JLD2
using Plots
using Statistics, LinearAlgebra
using LaTeXStrings

Base.ENV["GKSwstype"]="nul"

############################################################################
##  Load and Save the results
############################################################################
DIR = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02"

KP = jldopen(joinpath(DIR,"KP.jld2"))["KP"];



sol_KP = jldopen(joinpath(DIR,"sol_KP_3.jld2"))["sol_KP"]



############################################################################
##  Calculate the run simulation
############################################################################
scheme = ImplicitEM()#);linsolve = KernelCLFieldTheory.KrylovJL_GMRES())
RS_test = RunSetup(tspan=2000.0, 
                    tspan_thermalization=100.0, 
                    scheme=scheme, dtmax=5e-4, abstol=1e-4, reltol=1e-4,
                    NTr=1
)

KP = KernelCLFieldTheory.updateProblem(KP);

@time sol_KP = run_simulation(KP,RS_test);
#@time sol_KP = run_simulation(KP,RS_test;T=Float32,gpu=true);
fig_SK_KP = plotSKContour(KP,sol_KP)
fig_FW_KP = plotFWContour(KP,sol_KP)

obs_KP = KernelCLFieldTheory.calc_obs(KP,_sol_KP)


jldsave(joinpath(DIR,"sol_KP_dtmax_1em2.jld2"), sol_KP=sol_KP)
jldsave(joinpath(DIR,"obs_KP_dtmax_1em2.jld2"), obs_KP=obs_KP)

KernelCLFieldTheory.savefig(fig_SK_KP,joinpath(DIR,"fig_SK_KP_dtmax_1em2.pdf"))
KernelCLFieldTheory.savefig(fig_FW_KP,joinpath(DIR,"fig_FW_KP_dtmax_1em2.pdf"))




############################################################################
##  Calculate the boundary terms
############################################################################
Ys = Float64.(collect(1:0.2:20))
BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=collect(0:0.1:8))
@time B1_Omega, B1_Y, all_trVs = KernelCLFieldTheory.calcBoundaryTerms3(sol_KP,BT)#, all_trVs = all_trVs)

all_trVs_5em4_1em4 = all_trVs
B1_Omega_5em4_1em4 = B1_Omega


all_trVs_1em3 = all_trVs
B1_Omega_1em3 = B1_Omega

all_trVs_1em3_1em3 = all_trVs
B1_Omega_1em3_1em3 = B1_Omega

all_trVs_1em2 = all_trVs
B1_Omega_1em2 = B1_Omega

all_trVs_1em2_1em2 = all_trVs
B1_Omega_1em2_1em2 = B1_Omega


all_trVs_1em1 = all_trVs
B1_Omega_1em1 = B1_Omega

all_trVs_1em1_1em1 = all_trVs
B1_Omega_1em1_1em1 = B1_Omega


begin
    B1 = B1_Omega

    observables = ["xRe", "xIm", "x2Re", "x2Im", "corr0tRe", "corr0tIm"]
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

jldsave(joinpath(DIR,"BTs_dtmax_5em4_1em4.jld2"), BT=BT)
jldsave(joinpath(DIR,"B1_Omega_dtmax_5em4_1em4.jld2"), B1_Omega=B1_Omega_5em4_1em4)
#jldsave(joinpath(DIR,"B1_Y_dtmax_1em1.jld2"), B1_Y=B1_Y)
jldsave(joinpath(DIR,"all_trVs_dtmax_5em4_1em4.jld2"), all_trVs=all_trVs_5em4_1em4)

KernelCLFieldTheory.savefig(fig_BT_Omega,joinpath(DIR,"fig_BT_Omega_dtmax_1em1.pdf"))
KernelCLFieldTheory.savefig(fig_BT_Y,joinpath(DIR,"fig_BT_Y_dtmax_1em1.pdf"))














using Measurements

dtmaxes = [1e-1,1e-2,1e-3,5e-4]
BTs = [0.0583 ± 0.0031,0.061 ± 0.003,0.0099 ± 0.0017]
BTs = [0.21 ± 0.0012, 0.065 ± 0.0031,0.004 ± 0.0061,0.0018 ± 0.0068]
BTs = [0.21, 0.065,0.004,0.0018]
fig_BTextrap = scatter(dtmaxes,BTs;yaxis=:log,xaxis=:log,ylim=[1e-5,1],ylabel=L"B_1", xlabel=L"dt^{\max}",label=false, KernelCLFieldTheory.plot_setup()...)

KernelCLFieldTheory.savefig(fig_BTextrap,joinpath(DIR,"fig_BTextrap_2000.pdf"))

