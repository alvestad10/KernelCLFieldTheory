using KernelCLFieldTheory
using JLD2
using Plots
using LaTeXStrings
using Statistics, LinearAlgebra


############################################################################
##  Load and Save the results
############################################################################
DIR = "PaperPlots/CanonicalSK/results/RT_10_at_01_as_02/run1"
if !isdir(DIR)
    mkpath(DIR)
end
############################################################################


Base.ENV["GKSwstype"]="nul"

M = ScalarField_from_at(D=1,m=1.0,λ=1.,RT=1.0,β=0.4,at=0.1,n_steps=8,as=0.2,
                        Δβ = 0.5                
                        #ΔE = 0.05              
)

KP = KernelCLFieldTheory.KernelProblem(M);


######
# Starting from 
####
# DIR = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02/"
# KP = jldopen(joinpath(DIR,"KP_2.jld2"))["KP"]
# lhistory = jldopen(joinpath(DIR,"lhistory_2.jld2"))
# lhistory_train = lhistory["lhistory_train"]
# lhistory_val = lhistory["lhistory_val"]
#spy(KP.kernel.H[1:div(end,2),:] .- diag(KP.kernel.H))
#K = KernelCLFieldTheory.getK(KP.kernel.H)
#spy(K[2])
####


############################################################################
#    RunSetup
#
#    Contains the parameters for the simulation
#
#    Under is the changes from the default
#
#    Note: The termalization is removed is u0 initialized from end of 
#    previous run
############################################################################
scheme = ImplicitEM()
RS_train = RunSetup(tspan=3,
                    saveat=0.005,
                    tspan_thermalization=50,  
                    dtmax=1e-4, abstol=1e-4, reltol=1e-4,
                    scheme=scheme
)

# The validation setup
RS_val = RunSetup(tspan=1, 
                    tspan_thermalization=1, 
                    scheme=scheme
)

# The testing setup
RS_test = RunSetup(tspan=1000, 
                    tspan_thermalization=100, 
                    scheme=scheme, dtmax=1e-4, abstol=1e-4, reltol=1e-4,
                    NTr=10
)


############################################################################
### Initial testing
############################################################################
#@time sol_noKernel = run_simulation(KP,RS_test);

#fig_SK_noKernel = plotSKContour(KP,sol_noKernel)
#fig_FW_noKernel = plotFWContour(KP,sol_noKernel)

#obs_noKernel = KernelCLFieldTheory.calc_obs(KP,sol_noKernel)




############################################################################
#    Kernel Optimization setup
############################################################################
opt = ADAM(0.001, (0.5, 0.9))
epochs = 10
runs_pr_epoch = 50
n_gradient_pr_run = 1


############################################################################
#    Setting up the callback function
#
#   The callback function is called after each iteration of the optimization
#   algorithm. It is used to monitor the progress of the optimization. 
#
#   The Callback function cb can be changed to suit the needs of the user.
#   
#   The callback function stores intermediate results in the lhistory_train
#   and lhistory_val dictionaries.
############################################################################
lhistory_val = Dict(:L => [], :LSym => [], :evalsK => [], :detK => [], :symK => [])
lhistory_train = Dict(:L => [], :LSym => [], :evalsK => [], :detK => [], :symK => [])

cb(sol,KP::KernelProblem;type="val",show_plot=false, verbose=true) = begin

    if show_plot
        display(plotSKContour(KP,sol))
    end

    LTrain = mean(KernelCLFieldTheory.calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
    LSym =  KernelCLFieldTheory.calcSymLoss(sol,KP)

    KRe, KIm = KernelCLFieldTheory.getK(KP.kernel)
    KC = KRe .+ im * KIm
    evals = eigvals(KC)

    if verbose
        print("  LTrain: ", round(LTrain,digits=5), ", LSym: ", round(LSym,digits=5))#, ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5),", LCorr: ", round(LCorr,digits=5))
    end

    lhistory = lhistory_val
    if type == "train"
        lhistory = lhistory_train
    end

    if !isnothing(lhistory)
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LSym],LSym)
        push!(lhistory[:evalsK],evals)
        append!(lhistory[:detK],det(KC))
        append!(lhistory[:symK],norm(KC .- transpose(KC)))
    end
    return LSym
end



############################################################################
#    Optimization
############################################################################
LK = LearnKernel(KP,RS_train;RS_val=RS_val, 
                 opt=opt, epochs=epochs, 
                 runs_pr_epoch=runs_pr_epoch,
                 n_gradient_pr_run = n_gradient_pr_run,
                 cb=cb)
bestKernel, bestl, KP, u0s = learnKernel(LK; reset_u0s=true, validation=false)

########## SAVE
jldsave(joinpath(DIR,"KP.jld2"), KP=KP)

############################################################################
#    Plot the loss history
############################################################################
lplot_train = plot(lhistory_train[:L],label="L", yaxis=:log)
lsymplot_train = plot(lhistory_train[:LSym],label="LSym", yaxis=:log)

#lplot_val = plot(lhistory_val[:L],label="L", yaxis=:log)
#lsymplot_val = plot(lhistory_val[:LSym],label="LSym", yaxis=:log)


########## SAVE
jldsave(joinpath(DIR,"lhistory.jld2"), lhistory_train=lhistory_train, lhistory_val=lhistory_val)
KernelCLFieldTheory.savefig(lplot_train,joinpath(DIR,"lplot_train.pdf"))
KernelCLFieldTheory.savefig(lsymplot_train,joinpath(DIR,"lsymplot_train.pdf"))
#KernelCLFieldTheory.savefig(lplot_val,joinpath(DIR,"lplot_val.pdf"))
#KernelCLFieldTheory.savefig(lsymplot_val,joinpath(DIR,"lsymplot_val.pdf"))

############################################################################
#    Testing optimized kernel
############################################################################
# bestKP = KernelCLFieldTheory.KernelProblem(M, kernel=bestKernel);
# @time sol_bestKP = run_simulation(bestKP,RS_test; u0=u0s);

# fig_SK_bestKP = plotSKContour(bestKP,sol_bestKP)
# fig_FW_bestKP = plotFWContour(bestKP,sol_bestKP)

# obs_bestKP = KernelCLFieldTheory.calc_obs(KP,sol_bestKP)

#u0s = [tr[:,end] for tr in sol_KP]
@time sol_KP = run_simulation(KP,RS_test)#; u0=u0s);
fig_SK_KP = plotSKContour(KP,sol_KP)
fig_FW_KP = plotFWContour(KP,sol_KP)

obs_KP = KernelCLFieldTheory.calc_obs(KP,sol_KP)

# @time sol_new_u0Init = run_simulation(bestKP,RS_test);

# fig_SK_new_u0Init = plotSKContour(bestKP,sol_new_u0Init)
# fig_FW_new_u0Init = plotFWContour(bestKP,sol_new_u0Init)

# obs_new_u0Init = KernelCLFieldTheory.calc_obs(bestKP,sol_new_u0Init)


########## SAVE
jldsave(joinpath(DIR,"sol_KP.jld2"), sol_KP=sol_KP)
jldsave(joinpath(DIR,"obs_KP.jld2"), obs_KP=obs_KP)

KernelCLFieldTheory.savefig(fig_SK_KP,joinpath(DIR,"fig_SK_KP.pdf"))
KernelCLFieldTheory.savefig(fig_FW_KP,joinpath(DIR,"fig_FW_KP.pdf"))







############################################################################
#    Boundary Terms
############################################################################


Ys = collect(0:0.2:15)
BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=collect(0:0.1:8))
@time B1 = calcBoundaryTerms(sol_KP,BT)

begin
    observables = ["xRe", "xIm", "x2Re", "x2Im", "corr0tRe", "corr0tIm"]
    t_steps = KP.model.contour.t_steps

    fig_BT = plot(xlabel=L"\Omega", ylabel=L"B_1(\Omega)";KernelCLFieldTheory.plot_setup(:topright)...)
    for i in eachindex(observables)
        if i > 4 || i < 3
            continue
        end
        _BT_mean = vec(mean(B1[(i-1)*t_steps .+ (1:t_steps),:],dims=1))
        scatter!(fig_BT,Ys,_BT_mean, label=observables[i];KernelCLFieldTheory.markers_dict(i)...)
    end
    hline!(fig_BT,[0.0];label=false,KernelCLFieldTheory.solution_line_dict()...)
    fig_BT
end

jldsave(joinpath(DIR,"BTs.jld2"), BT=BT)
jldsave(joinpath(DIR,"B1.jld2"), B1=B1)

KernelCLFieldTheory.savefig(fig_BT,joinpath(DIR,"fig_BT.pdf"))



############################################################################

### SAVE
# jldsave(joinpath(DIR,"lhistory.jld2"), lhistory_train=lhistory_train, lhistory_val=lhistory_val)
# KernelCLFieldTheory.savefig(lplot_train,joinpath(DIR,"lplot_train.pdf"))
# KernelCLFieldTheory.savefig(lsymplot_train,joinpath(DIR,"lsymplot_train.pdf"))
# KernelCLFieldTheory.savefig(lplot_val,joinpath(DIR,"lplot_val.pdf"))
# KernelCLFieldTheory.savefig(lsymplot_val,joinpath(DIR,"lsymplot_val.pdf"))

# jldsave(joinpath(DIR,"sol_bestKP.jld2"), sol_bestKP=sol_bestKP)
# jldsave(joinpath(DIR,"obs_bestKP.jld2"), obs_bestKP=obs_bestKP)
# jldsave(joinpath(DIR,"bestKP.jld2"), bestKP=bestKP)

# KernelCLFieldTheory.savefig(fig_SK_bestKP,joinpath(DIR,"fig_SK_bestKP.pdf"))
# KernelCLFieldTheory.savefig(fig_FW_bestKP,joinpath(DIR,"fig_FW_bestKP.pdf"))


# jldsave(joinpath(DIR,"sol_KP.jld2"), sol_KP=sol_KP)
# jldsave(joinpath(DIR,"obs_KP.jld2"), obs_KP=obs_KP)
# jldsave(joinpath(DIR,"KP.jld2"), KP=KP)

# KernelCLFieldTheory.savefig(fig_SK_KP,joinpath(DIR,"fig_SK_KP.pdf"))
# KernelCLFieldTheory.savefig(fig_FW_KP,joinpath(DIR,"fig_FW_KP.pdf"))







#### LOAD
# KP = jldopen(joinpath(DIR,"bestKP.jld2"),"r") do file
#     kernel = file["bestKP"].kernel
#     model = file["bestKP"].model
#     return KernelCLFieldTheory.KernelProblem(model, kernel=kernel)
# end
