export RunSetup, run_simulation
export LambaEM, ImplicitEM

"""
    Parameters for the simulation
"""
@with_kw mutable struct RunSetup
    tspan=20
    NTr = 1
    saveat=0.01
    scheme=ImplicitEM() 
    dt=1e-4
    abstol=1e-3
    reltol=1e-3
    dtmax=1e-3
    adaptive=true
    tspan_thermalization=5
    dt_thermalization = 1e-3
    scheme_thermalization=ImplicitEM()
    dtmax_thermalization=1e-1
    abstol_thermalization=1e-2
    reltol_thermalization=1e-2
end



function run_simulation(KP::KernelProblem, runSetup::RunSetup; seed=nothing, u0 = nothing, T=Float64)
    @unpack tspan, NTr, saveat,scheme, dt, abstol, reltol, dtmax, adaptive, tspan_thermalization = runSetup
    
    @unpack kernel, a, b, model = KP
    @unpack t_steps = model.contour
    

    if isnothing(u0)
        u0 = [get_initial_value(model,T) for i in 1:NTr]
    else
        tspan_thermalization = 0.01
    end

    noise_rate_prototype = kernel.H #get_noise_rate_prototype(model,T)
    caches = [get_caches(model, T) for _ in 1:NTr]

    function prob_func(prob,i,repeat)
        if isnothing(seed)
            prob = remake(prob, p=caches[i], u0=u0[i])
        else
            prob = remake(prob, seed=seed + i, p = caches[i], u0=u0[i])
        end
    end
    
    isIdentity = all(diag(kernel.H) .- 1. .== 0.0)
    if isIdentity
        prob = SDEProblem(a,b,u0[1],(0.0,tspan+tspan_thermalization),caches[1])
    else
        prob = SDEProblem(a,b,u0[1],(0.0,tspan+tspan_thermalization),caches[1],
                                    noise_rate_prototype=noise_rate_prototype
                )
    end

    ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)

    # Getting progressbar
    global_logger(TerminalLogger())

    return solve(ensemble_prob,
                    #prob,
                    scheme,
                    EnsembleThreads(), trajectories=NTr,
                    adaptive = adaptive,
                    dt = dt,
                    maxiters = 1e8,
                    progress=true,
                    progress_steps = 100,
                    saveat=tspan_thermalization:saveat:tspan+tspan_thermalization,
                    save_start = true,
                    dtmax = dtmax,
                    abstol = abstol,reltol = reltol)

end
