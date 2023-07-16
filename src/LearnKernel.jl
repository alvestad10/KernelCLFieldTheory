export LearnKernel, learnKernel

function learnKernel(KP::KernelProblem, RS::RunSetup; RS_val::RunSetup=RS, cb=(KP::KernelProblem, sol=nothing, addtohistory=false) -> (return 0))

    @unpack kernel = KP

    trun = @elapsed sol = run_simulation(KP,RS_val)
    LD = mean(calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
    l = cb(KP; sol=sol, addtohistory=true)

    bestKernel = deepcopy(KP.kernel)
    #bestLSym = l

    tdL = 0.

    epochs = 2
    runs_pr_epoch = 30

    bestLD = LD
    lastImprovement = 0


    opt = ADAM(0.001)

    for i in 1:epochs
        for j in 1:runs_pr_epoch
            u0s = [tr[:,end] for tr in sol]

            tdL = @elapsed dHs = getdH_imx(sol,KP)
            
            Flux.update!(opt, KP.kernel.H, dHs)

            KP = updateProblem(KP)

            trun = @elapsed sol = run_simulation(KP, RS; u0=u0s)

            LD = mean(calcIMXLoss(sol[tr],KP) for tr in eachindex(sol))
            println("EPOCH ", i, ", BATCH:", j," ------ LDrift=", round(LD,digits=5), "\t (time_grad: ", round(tdL,digits=2), ")")

            if LD < bestLD
                bestLD = LD
                bestKernel = deepcopy(KP.kernel)
                lastImprovement = 0
            end

            if lastImprovement > 10
                break
            else
                lastImprovement += 1
            end


        end

        trun = @elapsed sol = run_simulation(KP,RS_val)
        l = cb(KP; sol = sol, addtohistory=true)
    end

    return bestKernel
end