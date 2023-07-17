export LearnKernel, learnKernel

function learnKernel(KP::KernelProblem, RS::RunSetup; RS_val::RunSetup=RS, cb=(KP::KernelProblem; kwargs...) -> (return 0), lhistory_train=nothing, lhistory_val=nothing)

    @unpack kernel = KP

    trun = @elapsed sol = run_simulation(KP,RS_val)
    LD = mean(calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
    l = cb(KP; sol=sol, lhistory=lhistory_val, show_plot=true)

    bestKernel = deepcopy(KP.kernel)
    #bestLSym = l

    tdL = 0.

    epochs = 2
    runs_pr_epoch = 50

    bestLD = LD


    opt = ADAM(0.001)

    for i in 1:epochs
        

        for j in 1:runs_pr_epoch
            u0s = [tr[:,end] for tr in sol]
            trun = @elapsed sol = run_simulation(KP, RS; u0=u0s)
            l = cb(KP; sol = sol, lhistory=lhistory_train, show_plot=false)

            tdL = @elapsed dHs = getdH_imx(sol,KP)
            
            Flux.update!(opt, KP.kernel.H, dHs)

            # Updating the drift and noise term with the new kernel
            KP = updateProblem(KP)

            LD = mean(calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
            println("EPOCH ", i, ", BATCH:", j," ------ LDrift=", round(LD,digits=5), "\t (time_grad: ", round(tdL,digits=2), ")")

            if LD < bestLD
                bestLD = LD
                bestKernel = deepcopy(KP.kernel)
            end


        end

        trun = @elapsed sol = run_simulation(KP,RS_val)
        l = cb(KP; sol = sol, lhistory=lhistory_val, show_plot=true)
    end

    return bestKernel
end