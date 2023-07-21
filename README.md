# KernelCLFieldTheory


To run the code you need a version of Julia installed, then you can make separate scripts or follow the RealTimeScalaField.jl file which can be run in bash using `julia --project=. RealTimeScalaField.jl` or line by line using the Julia vscode extension. Before you run the code follow the Instantite section below to setup the necessary packages.

TODO: At the moment RealTimeScalaField.jl does not save figures or configurations.

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
First a simple example to run the 1+1D Field theory on the split Schwinger-Keldysh contour (Alexandru2016) without a kernel up to 1.0 in real-time, and then plotting the result. (This is the content of )

```julia
using KernelCLFieldTheory
M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.0,β=0.4,at=0.2,n_steps=8,as=0.2,
                        Δβ = 0.5                
                        #ΔE = 0.0              
)

KP = KernelCLFieldTheory.KernelProblem(M);
RS = RunSetup(tspan=20, 
                    tspan_thermalization=1000,  
                    scheme=LambaEM()
)

@time sol = run_simulation(KP,RS);
plotSKContour(KP,sol)
plotFWContour(KP,sol)
```


## Example 2

Second a simple example to learn a kernel for the 1+1D Field Theory on the split Schwinger-Keldysh contour (Alexandru2016) up to 1.6 in real-time

```julia
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

```


# How to update the code


