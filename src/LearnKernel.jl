export LearnKernel, learnKernel
export ADAM, Descent
struct LearnKernel
    KP::KernelProblem
    RS::RunSetup
    RS_val::RunSetup
    opt
    epochs
    runs_pr_epoch
    n_gradient_pr_run
    cb::Function
end

function LearnKernel(KP::KernelProblem, RS::RunSetup; RS_val::RunSetup=RS, opt=ADAM(0.001), epochs=5, runs_pr_epoch=10, n_gradient_pr_run=1, cb=(sol,KP::KernelProblem; kwargs...) -> (return 0))
    return LearnKernel(KP,RS,RS_val,opt,epochs, runs_pr_epoch, n_gradient_pr_run,cb)
end

function learnKernel(LK::LearnKernel; reset_u0s = false, val_seed=100, u0s=nothing)


    @unpack KP, RS, RS_val, epochs, runs_pr_epoch, n_gradient_pr_run, cb, opt = LK
    @unpack kernel = KP

    trun = @elapsed sol = run_simulation(KP,RS_val; seed=val_seed, u0s)
    
    println("  ")
    print(" ---------- INITIAL: ")
    l = cb(sol, KP; type="val", show_plot=true, verbose=true)
    println("\t\t(time_run: ", round(trun,digits=2))
    println("  ")

    bestKernel = deepcopy(KP.kernel)
    bestL = l

    u0s = [tr[:,end] for tr in sol]

    for i in 1:epochs
        println(" ---------- EPOCH ", i, " ---------- ")
        for j in 1:runs_pr_epoch
            trun = @elapsed sol = run_simulation(KP, RS; u0=u0s)#, seed=100)
            
            tdL = 0.
            for k in 1:n_gradient_pr_run

                tdL += @elapsed dHs = getdH_imx(sol,KP)
                Flux.update!(opt, KP.kernel.H, dHs)

                # Updating the drift and noise term with the new kernel
                KP = updateProblem(KP)

                #LD = mean(calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
                print("EPOCH ", i, ", BATCH: ", j, ".", k," ::::: ") #, LDrift=", round(LD,digits=5), "\t (time_grad: ", round(tdL,digits=2), ")")

                cb(sol, KP; type="train", show_plot=false, verbose=true)
            end
            tdL = tdL/n_gradient_pr_run
            println("\t\t(time_grad: ", round(tdL,digits=2), ")", "(time_run_train: ", round(trun,digits=2),")")

            u0s = [tr[:,end] for tr in sol]
        end

        if reset_u0s
            u0s = nothing
        end

        trun = @elapsed sol = run_simulation(KP,RS_val;u0=u0s,seed=val_seed)
        println(" ")
        print(" VALIDATION: ")
        l = cb(sol, KP; type="val", show_plot=true, verbose=true)
        println("\t\t(time_run_val: ", round(trun,digits=2), ")")
        println(" ")
        if l < bestL
            bestL = l
            bestKernel = deepcopy(KP.kernel)
        end
    end

    return bestKernel, bestL, KP, u0s
end