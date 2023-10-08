using KernelCLFieldTheory
using JLD2
using Plots
using Statistics, LinearAlgebra
using LaTeXStrings
using Measurements
using Random

Base.ENV["GKSwstype"]="nul"

Random.seed!(12345122343)

############################################################################
##  Load and Save the results
############################################################################
DIR = "PaperPlots/LatticePlots/results/RT_32/extrap_BT"
DIR_KP = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02"

KP = jldopen(joinpath(DIR_KP,"KP_3.jld2"))["KP"];
KP = KernelCLFieldTheory.updateProblem(KP);


tols = [1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4]

for tol in tols
    println("----------------------------------------------")
    println("----------------------------------------------")
    println("------ tol = ",tol," -------------------------")
    println("----------------------------------------------")
    println("----------------------------------------------")

    scheme = ImplicitEM()#);linsolve = KernelCLFieldTheory.KrylovJL_GMRES())
    RS_test = RunSetup(tspan=100.0,
                        tspan_thermalization=10.0, 
                        #saveat=1.0,
                        scheme=scheme, dtmax=tol, abstol=tol, reltol=tol,
                        NTr=1
    )


    @time sol_KP = run_simulation(KP,RS_test; seed=345342243);
    jldsave(joinpath(DIR,string("sol_",tol,".jld2")), sol_KP=sol_KP)

    obs_KP = KernelCLFieldTheory.calc_obs(KP,sol_KP);
    obs_KP = [obs_KP[i] .± obs_KP[i+1] for i in 1:2:length(obs_KP)];
    fig_SK_KP = plotSKContour(KP,sol_KP; obs = obs_KP)
    fig_FW_KP = plotFWContour(KP,sol_KP; obs = obs_KP)

    jldsave(joinpath(DIR,string("obs_",tol,".jld2")), obs_KP=obs_KP)

    KernelCLFieldTheory.savefig(fig_SK_KP,joinpath(DIR,string("fig_SK_",tol,".pdf")))
    KernelCLFieldTheory.savefig(fig_FW_KP,joinpath(DIR,string("fig_FW_",tol,".pdf")))



    Ys = Float64.(collect(0:0.2:20))
    BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=collect(0:0.1:8))
    @time B1_Omega, B1_Y, trVs = KernelCLFieldTheory.calcBoundaryTerms(sol_KP,BT)#, trVs = trVs)

    jldsave(joinpath(DIR,string("trVs_",tol,".jld2")), trVs=trVs)

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


    jldsave(joinpath(DIR,string("BTs_",tol,".jld2")), BT=BT)
    jldsave(joinpath(DIR,string("B1_Omega_",tol,".jld2")), B1_Omega=B1_Omega)
    jldsave(joinpath(DIR,string("B1_Y_",tol,".jld2")), B1_Y=B1_Y)

    KernelCLFieldTheory.savefig(fig_BT_Omega,joinpath(DIR,string("fig_BT_Omega_",tol,".pdf")))
    KernelCLFieldTheory.savefig(fig_BT_Y,joinpath(DIR,string("fig_BT_Y_",tol,".pdf")))
end