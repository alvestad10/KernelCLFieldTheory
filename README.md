# KernelCLFieldTheory


To run the code you need a version of Julia installed, then you can make separate scripts or follow the RunScalarFieldOptimization.jl file which can be run in bash using `julia --project=. RunScalarFieldOptimization.jl` or line by line using the Julia vscode extension. Before you run the code follow the Instantite section below to setup the necessary packages.

## Instantiate

To start out with this project you need a version of julia installed and a cloned version of this repository. Before we can run the code we need to instantiate the Julia environment for the packages. To this we need to initialize the project run these lines inside the Julia REPL (From inside the project directory)
```julia
    import Pkg
    Pkg.activate(".")
    Pkg.instantiate()
```
For more information see: https://docs.julialang.org/en/v1/stdlib/Pkg/

Now all dependencies should be downloaded and the code is ready to be run. Remember that if one starte Julia from the terminal, you need to start it using `julia --project=.` when inside the root directory of the repository.

The project is setup as a package, which means that the general structure contains a `src/` folder which contains the module `KernelCLFieldThoery` which we need to load to use the componenets, then a Project.toml which contains all the packages used, and then some run files in the root folder. These can be altered for the process to be run. 

## Main parts of the project

The project have 4 main components;
- Model; implementation can be found in `src/Model.jl` which contains the model structure. Part of the model is also the contour. 
  - The ScalaField struct is defined such that we pass in the dimension D, and all the other parameters. The default is: `ScalarField_from_at(;D=1,m=1.,λ=0.1,RT=1.6,β=0.4,at=0.2,n_steps=8,as=0.2,kwargs...)`
  - In the kwargs on can pass the contour parameters. Default here is the canonical contour, while Δβ = 0.5 must be used for the split SK contour (Alexandru2016) and ΔE = 0.0 for tilted SK contour (or larger if also forward tilt)
- Kernel (`src/Kernel.jl`); implements a structure for the kernel we use. Now this is essentially only a matrix H, and the corresponding  funciton to get K.
- KernelProblem; this is to make a type putting together the model with a kernel, drift term and noise term. This is convenient as we can just pass this to the run_simulation function since this contains all necessary information to run the simulation.
- Finally we have the LearnKernel component, which takes the KernelProblem as input, and thenrun the kernel optimization algorithm. 


These components can be used in scripts to simulate what we need. Under us some examples how to use them


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

Second a simple example to learn a kernel for the 1+1D Field Theory on the split Schwinger-Keldysh contour (Alexandru2016) up to 1.0 in real-time

```julia
using KernelCLFieldTheory
using Plots
using Zygote
using Measurements
using Statistics
using LinearAlgebra


M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.0,β=0.4,at=0.2,n_steps=8,as=0.2,
                        Δβ = 0.5                
                        #ΔE = 0.0              
)

KP = KernelCLFieldTheory.KernelProblem(M);
RS_train = RunSetup(NTr=1, tspan=20, saveat=0.1, dt=1e-4, dtmax=1e-3, adaptive=true)
RS_val = deepcopy(RS_train)
RS_val.tspan = 1000
RS_val.scheme = KernelCLFieldTheory.LambaEM()

lhistory = Dict(:L => [], :LSym => [], :evalsK => [], :detK => [], :symK => [])

cb(KP::KernelProblem;sol=sol,addtohistory=false) = begin

    display(plotSKContour(KP,sol))

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
```

Now we can use the optimal kernel and run once more with a higher statistics
```julia
RS_val2 = RunSetup(NTr=1, tspan=5000, saveat=0.01, dt=1e-4, dtmax=1e-3, adaptive=true)
@time sol = run_simulation(bestKP,RS_val2);
plotSKContour(bestKP,sol)
```


# How to update the code

At the moment the learnKernel function is hardcoded with values for the optimization. (Have not got time yet to update this)

This is suppost to be the main loop of the optimizaiton process, can be tuned for different strategies. 


The structure of the code is as follows:
The model, including the contour, can be made by `ScalarField` or `ScalarField_from_at`. See the example of parameters. The contour is specified with, nothing extra: Canonical SK, Δβ = 0.5: split SK contour (Alexandru2016) and ΔE = 0.0 for tilted SK contour (or larger if also forward tilt). 

The specific implementation of the discretized action is in the Implementations folder, where also the loss funciton is located.
In this function it is possible to change the number of steps forward for each configuration when calculating the loss function (and its gradient).




