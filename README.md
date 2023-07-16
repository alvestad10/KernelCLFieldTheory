# KernelCLFieldTheory


To run the code you need a version of Julia installed, then you can make separate scripts or follow the RunScalarFieldOptimization.jl file which can be run in bash using `julia --project=. RunScalarFieldOptimization.jl` or line by line using the Julia vscode extension. Before you run the code follow the Instantite section below to setup the necessary packages.

## Instantiate

To initialize the project run these comments inside the Julia REPL (From inside the project directory)
```julia
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()
```
For more information see: https://docs.julialang.org/en/v1/stdlib/Pkg/

Now all dependencies should be downloaded and the code is ready to be run.

## Example 1
First a simple example to run the 1+1D Field theory on the split Schwinger-Keldysh contour (Alexandru2016) without a kernel up to 0.6 in real-time, and then plotting the result

```julia
using KernelCLFieldTheory

M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=0.6,β=0.4,at=0.2,n_steps=8,as=0.2,
                        Δβ = 0.5                
)
KP = KernelCLFieldTheory.KernelProblem(M)
RS = RunSetup(NTr=1, tspan=1000, saveat=0.01, dt=1e-4, dtmax=1e-3, adaptive=true)

@time sol = run_simulation(KP,RS);

plotSKContour(KP,sol)
```


## Example 2

Second a simple example to learn a kernel for the 1+1D Field Theory on the split Schwinger-Keldysh contour (Alexandru2016) up to 1.6 in real-time

```julia
using KernelCLFieldTheory
using Plots
using Zygote
using Measurements
using Statistics
using JLD2


M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.6,β=0.4,at=0.2,n_steps=8,as=0.2,
                        Δβ = 0.5                
)

KP = KernelCLFieldTheory.KernelProblem(M)
RS_train = RunSetup(NTr=1, tspan=100, saveat=0.01, dt=1e-4, dtmax=1e-3, adaptive=true)
RS_val = deepcopy(RS_train)
RS_val.tspan = 500

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

bestKernel = learnKernel(KP,RS_train; cb=cb)

bestKP = KernelCLFieldTheory.KernelProblem(M, kernel=bestKernel)
@time sol = run_simulation(KP,RS_val);

plotSKContour(KP,sol)
```

Now we can use the optimal kernel and run once more with a higher statistics
```julia
RS_train = RunSetup(NTr=1, tspan=2000, saveat=0.01, dt=1e-4, dtmax=1e-3, adaptive=true)
@time sol = run_simulation(bestKP,RS_val);
plotSKContour(bestKP,sol)
```


# How to update the code

At the moment the learnKernel function is hardcoded with values for the optimization. (Have not got time yet to update this)

This is suppost to be the main loop of the optimizaiton process, can be tuned for different strategies. 


The structure of the code is as follows:
The model, including the contour, can be made by `ScalarField` or `ScalarField_from_at`. See the example of parameters. The contour is specified with, nothing extra: Canonical SK, Δβ = 0.5: split SK contour (Alexandru2016) and ΔE = 0.0 for tilted SK contour (or larger if also forward tilt). 

The specific implementation of the discretized action is in the Implementations folder, where also the loss funciton is located.
In this function it is possible to change the number of steps forward for each configuration when calculating the loss function (and its gradient).




