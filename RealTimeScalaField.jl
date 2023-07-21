using KernelCLFieldTheory
using LinearAlgebra, Statistics


M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.6,β=0.4,at=0.2,n_steps=8,as=0.2,
                        Δβ = 0.5                
                        #ΔE = 0.0              
)

KP = KernelCLFieldTheory.KernelProblem(M);

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
RS_train = RunSetup(tspan=20, 
                    tspan_thermalization=1000,  
                    scheme=LambaEM()
)

# The validation setup
RS_val = RunSetup(tspan=100, 
                    tspan_thermalization=1000, 
                    scheme=LambaEM()
)

# The testing setup
RS_test = RunSetup(tspan=5000, 
                    tspan_thermalization=1000, 
                    scheme=LambaEM()
)


############################################################################
#    Kernel Optimization setup
############################################################################
opt = ADAM(0.001, (0.5, 0.9))
epochs = 5
runs_pr_epoch = 5



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
                 cb=cb)
bestKernel, bestl, KP, u0s = learnKernel(LK; reset_u0s=false)

####
# Updating LK with new KP just construct a new LK above.
####


############################################################################
#    Testing optimized kernel
############################################################################
bestKP = KernelCLFieldTheory.KernelProblem(M, kernel=bestKernel)
@time sol = run_simulation(bestKP,RS_test; u0=u0s);

plotSKContour(bestKP,sol)
plotFWContour(bestKP,sol)


############################################################################
#    Plot the loss history
############################################################################
plot(lhistory_train[:L],label="L", yaxis=:log)
plot(lhistory_train[:LSym],label="LSym", yaxis=:log)

plot(lhistory_val[:L],label="L", yaxis=:log)
plot(lhistory_val[:LSym],label="LSym", yaxis=:log)