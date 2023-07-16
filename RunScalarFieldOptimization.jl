using KernelCLFieldTheory
using Plots
using Zygote
using Measurements
using Statistics
using JLD2


M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.6,β=0.4,at=0.2,n_steps=8,as=0.2,
                        #Δβ = 0.5                
)

KP = KernelCLFieldTheory.KernelProblem(M)
RS_train = RunSetup(NTr=1, tspan=10, saveat=0.01, dt=1e-4, dtmax=1e-3, adaptive=true)
RS_val = deepcopy(RS_train)
RS_val.tspan = 100

@time sol = run_simulation(KP,RS_val);
plotSKContour(KP,sol)

lhistory = Dict(:L => [], :LSym => [], :evalsK => [], :detK => [], :symK => [])

cb(KP::KernelProblem;sol=sol,addtohistory=false) = begin

    LTrain = mean(KernelCLFieldTheory.calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
    LSym =  KernelCLFieldTheory.calcSymLoss(sol,KP)

    KRe, KIm = KernelCLFieldTheory.getK(KP.kernel)
    KC = KRe .+ im * KIm
    evals = eigvals(KC)


    println("LTrain: ", round(LTrain,digits=5), ", LSym: ", round(LSym,digits=5))#, ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5),", LCorr: ", round(LCorr,digits=5))

    if addtohistory
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LSym],LSym)
        push!(lhistory[:evalsK],evals)
        append!(lhistory[:detK],det(KC))
        append!(lhistory[:symK],norm(KC .- transpose(KC)))
    end
    return LSym
end

bestKernel = learnKernel(KP,RS_train; RS_val=RS_val, cb=cb)

bestKP = KernelCLFieldTheory.KernelProblem(M, kernel=bestKernel)
@time sol = run_simulation(bestKP,RS_val);

plotSKContour(KP,sol)
