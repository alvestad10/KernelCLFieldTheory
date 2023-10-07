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



function run_simulation(KP::KernelProblem, runSetup::RunSetup; seed=nothing, u0 = nothing, T=Float64, gpu=false)
    @unpack tspan, NTr, saveat,scheme, dt, abstol, reltol, dtmax, adaptive, tspan_thermalization = runSetup
    
    @unpack kernel, a, b, model = KP
    @unpack t_steps = model.contour
    

    if isnothing(u0)
        u0 = [get_initial_value(model,T) for i in 1:NTr]
    else
        tspan_thermalization = 0.01
    end

    noise_rate_prototype = get_noise_rate_prototype(model,T)
   
    if gpu
        u0 = [cu(u0[i]) for i in 1:NTr]
        noise_rate_prototype = cu(noise_rate_prototype)
    end
   
    caches = [get_caches(model, T; gpu=gpu) for _ in 1:NTr]

    function prob_func(prob,i,repeat)
        if isnothing(seed)
            prob = remake(prob, p=caches[i], u0=u0[i])
        else
            prob = remake(prob, seed=seed + i, p = caches[i], u0=u0[i])
        end
    end
    
    isIdentity = all(diag(kernel.H) .- 1. .== 0.0)
    if isIdentity
        
        # Getting sparsity pattern
        du0 = copy(u0[1])
        @time jac_sparsity = Symbolics.jacobian_sparsity((du, u) -> a(du, u, caches[1], 0.0),du0, u0[1])

        f = SDEFunction(a,b; jac_prototype = float.(jac_sparsity))
        prob = SDEProblem(f,b,u0[1],(0.0,tspan+tspan_thermalization),caches[1])
        
        #prob = SDEProblem(a,b,u0[1],(0.0,tspan+tspan_thermalization),caches[1])
    else
        prob = SDEProblem(a,b,u0[1],(0.0,tspan+tspan_thermalization),caches[1],
                                    noise_rate_prototype=noise_rate_prototype
                )
    end

    run_settings = Dict(
        :adaptive => adaptive,
        :dt => dt,
        :maxiters => 1e8,
        :progress => true,
        :progress_steps => 100,
        :saveat => tspan_thermalization:saveat:tspan+tspan_thermalization,
        :save_start => true,
        :dtmax => dtmax,
        :abstol => abstol,
        :reltol => reltol
    )


    # Getting progressbar
    global_logger(TerminalLogger())
    if NTr > 1
        ensprob = EnsembleProblem(prob,prob_func=prob_func)
        return solve(ensprob,
                scheme,
                EnsembleThreads(); 
                trajectories=NTr,
                run_settings...
                )
    else
        return solve(prob,
                    scheme;
                    run_settings...)
    end
    

    

end
