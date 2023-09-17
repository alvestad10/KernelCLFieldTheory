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

function learnKernel(LK::LearnKernel; reset_u0s = false, val_seed=100, u0s=nothing, validation=true)


    @unpack KP, RS, RS_val, epochs, runs_pr_epoch, n_gradient_pr_run, cb, opt = LK

    if validation
        trun = @elapsed sol = run_simulation(KP,RS_val; seed=val_seed, u0=u0s)
        println("  ")
        print(" ---------- INITIAL: ")
        l = cb(sol, KP; type="val", show_plot=true, verbose=true)
        println("\t\t(time_run: ", round(trun,digits=2))
        println("  ")
        bestL = l
        bestKernel = deepcopy(KP.kernel)
    else
        bestL = 0.
    end

    
    for i in 1:epochs

        if validation
            u0s = [tr[:,end] for tr in sol]
        end

        println(" ")
        println(" ---------- EPOCH ", i, " ---------- ")
        for j in 1:runs_pr_epoch
            trun = @elapsed sol = run_simulation(KP, RS; u0=u0s)#, seed=100)
            print("EPOCH ", i, ", BATCH: ", j, ".",)
            
            tdL = 0.
            for k in 1:n_gradient_pr_run
                print(" ",k)
                tdL += @elapsed dHs = getdH_imx(sol,KP)
                Flux.update!(opt, KP.kernel.H, dHs)

                # Updating the drift and noise term with the new kernel
                KP = updateProblem(KP)

                
                
                if k == n_gradient_pr_run
                    print(" ::::: ")
                    cb(sol, KP; type="train", show_plot=false, verbose=true)
                end
            end
            tdL = tdL/n_gradient_pr_run
            println("\t\t(time_grad: ", round(tdL,digits=2), ")", "(time_run_train: ", round(trun,digits=2),")")

            u0s = [tr[:,end] for tr in sol]
        end

        if reset_u0s || !validation
            u0s = nothing
        end

        if validation
            trun = @elapsed sol = run_simulation(KP,RS_val;u0=u0s,seed=val_seed)
            println(" ")
            print(" VALIDATION: ")
            l = cb(sol, KP; type="val", show_plot=true, verbose=true)
            println("\t\t(time_run_val: ", round(trun,digits=2), ")")
            if l < bestL
                bestL = l
                bestKernel = deepcopy(KP.kernel)
            end
        end
    end

    if validation
        return bestKernel, bestL, KP, u0s
    else
        return KP.kernel, 0., KP, u0s
    end
end