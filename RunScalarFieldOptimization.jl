using KernelCLFieldTheory
using Plots
using Zygote
using Measurements
using Statistics
using JLD2
using LinearAlgebra


M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.0,β=0.4,at=0.2,n_steps=8,as=0.2,
                        Δβ = 0.5                
                        #ΔE = 0.0              
)

KP = KernelCLFieldTheory.KernelProblem(M);
RS_train = RunSetup(NTr=1, tspan=20, saveat=0.1, dt=1e-4, dtmax=1e-3, adaptive=true)
RS_train.scheme = KernelCLFieldTheory.LambaEM()
RS_val = deepcopy(RS_train)
RS_val.tspan = 400
RS_val.scheme = KernelCLFieldTheory.LambaEM()

lhistory_val = Dict(:L => [], :LSym => [], :evalsK => [], :detK => [], :symK => [])
lhistory_train = Dict(:L => [], :LSym => [], :evalsK => [], :detK => [], :symK => [])

cb(KP::KernelProblem;sol=sol,lhistory=nothing,show_plot=false) = begin

    if show_plot
        display(plotSKContour(KP,sol))
    end

    LTrain = mean(KernelCLFieldTheory.calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
    LSym =  KernelCLFieldTheory.calcSymLoss(sol,KP)

    KRe, KIm = KernelCLFieldTheory.getK(KP.kernel)
    KC = KRe .+ im * KIm
    evals = eigvals(KC)


    println("LTrain: ", round(LTrain,digits=5), ", LSym: ", round(LSym,digits=5))#, ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5),", LCorr: ", round(LCorr,digits=5))

    if !isnothing(lhistory)
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LSym],LSym)
        push!(lhistory[:evalsK],evals)
        append!(lhistory[:detK],det(KC))
        append!(lhistory[:symK],norm(KC .- transpose(KC)))
    end
    return LSym
end

bestKernel = learnKernel(KP,RS_train; RS_val=RS_val, cb=cb, lhistory_train=lhistory_train, lhistory_val=lhistory_val)

RS_val2 = deepcopy(RS_train)
RS_val2.tspan = 5000
RS_val2.scheme = KernelCLFieldTheory.LambaEM()

bestKP = KernelCLFieldTheory.KernelProblem(M, kernel=bestKernel)
@time sol = run_simulation(bestKP,RS_val2);

plotSKContour(bestKP,sol)
plotFWContour(bestKP,sol)

plot(lhistory_train[:L],label="L")
plot(lhistory_train[:LSym],label="LSym")

plot(lhistory_train[:L],label="L")
plot(lhistory_train[:LSym],label="LSym")

