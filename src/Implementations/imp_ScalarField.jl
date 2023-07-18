
"""
    Calculate the observables for each of the trajectories
"""
function calc_obs(KP::KernelProblem{ScalarField{D}},sol;onlyCorr=false, max_inx = nothing) where {D}
    t_steps = KP.model.contour.t_steps
    n_steps = KP.model.n_steps

    T = eltype(sol[1]) #eltype( getKernelParams(KP.kernel) )
    
    if length(sol) == 1
        sol = sol[1]
    end

    if isnothing(max_inx)
        max_inx = length(sol)
    end

    if sol isa SciMLBase.RODESolution
        T = eltype(sol[1]) #eltype( getKernelParams(KP.kernel) )
        avgRe = zeros(T,t_steps)
        err_avgRe = zeros(T,t_steps)
        avgIm = zeros(T,t_steps)
        err_avgIm = zeros(T,t_steps)
        avg2Re = zeros(T,t_steps)
        err_avg2Re = zeros(T,t_steps)
        avg2Im = zeros(T,t_steps)
        err_avg2Im = zeros(T,t_steps)
        corr0tRe = zeros(T,t_steps)
        err_corr0tRe = zeros(T,t_steps)
        corr0tIm = zeros(T,t_steps)
        err_corr0tIm = zeros(T,t_steps)

        dt = sol.t[end] .- sol.t[end-1]
        nn = 1.0 / dt

        _u = Array(sol)[:,1:max_inx]
        _uRe = @view _u[1:t_steps*n_steps^D,:]
        _uIm = @view _u[t_steps*n_steps^D + 1:end,:]
        
        _uu = zeros(2t_steps,size(_u)[2])

        @inbounds for i in 1:n_steps
            for j in 1:n_steps^(D-1)
                _uu[1:t_steps,:] .+= @view _uRe[(j-1)*t_steps*n_steps + (i-1)*t_steps .+ (1:t_steps),:]
                _uu[t_steps+1:end,:] .+= @view _uIm[(j-1)*t_steps*n_steps + (i-1)*t_steps .+ (1:t_steps),:]
            end
        end

        phi_p_Re = (@view _uu[1:t_steps,:]) / n_steps^D #mean(_uu[1:t_steps,:,:,:],dims=(2,3))[:,1,1,:]
        phi_p_Im = (@view _uu[t_steps+1:end,:]) / n_steps^D #mean(_uu[t_steps+1:end,:,:,:],dims=(2,3))[:,1,1,:]

        phi_p_Re_1 = @view phi_p_Re[1,:]
        phi_p_Im_1 = @view phi_p_Im[1,:]
        N = length(phi_p_Re_1)

        bstrap = false

        for i in 1:t_steps
            S = MaximumEntropySampling(10)#BasicSampling
            X = zeros(N)
            phi_p_Re_i = phi_p_Re[i,:]
            phi_p_Im_i = phi_p_Im[i,:]
            
            if !onlyCorr
                avgRe[i] = mean(phi_p_Re_i)
                if bstrap
                    err_avgRe[i] = stderror(bootstrap(std, phi_p_Re_i, S))[1]#estimate(std,phi_p_Re_i)/sqrt(N)
                else
                    #err_avgRe[i] = estimate(std,phi_p_Re_i)/sqrt(N)
                    err_avgRe[i] = std(phi_p_Re_i)/sqrt(N/nn)
                end

                avgIm[i] = mean(phi_p_Im_i)
                if bstrap
                    err_avgIm[i] = stderror(bootstrap(std, phi_p_Im_i, S))[1]#estimate(std,phi_p_Im_i)/sqrt(N)
                else
                    #err_avgIm[i] = estimate(std,phi_p_Im_i)/sqrt(N)
                    err_avgIm[i] = std(phi_p_Im_i)/sqrt(N/nn)
                end            
                @. X = phi_p_Re_i^2 - phi_p_Im_i^2
                avg2Re[i] = mean(X)
                if bstrap    
                    err_avg2Re[i] = stderror(bootstrap(std, X, S))[1]#estimate(std,phi_p_Re_i.^2 .- phi_p_Im_i.^2)/sqrt(N)
                else
                    #err_avg2Re[i] = estimate(std,X)/sqrt(N)
                    err_avg2Re[i] = std(X)/sqrt(N/nn)
                end

                @. X = 2*phi_p_Re_i * phi_p_Im_i
                avg2Im[i] = mean(X)
                if bstrap
                    err_avg2Im[i] = stderror(bootstrap(std, X, S))[1]#estimate(std,2*phi_p_Re_i .* phi_p_Im_i)/sqrt(N)
                else
                    #err_avg2Im[i] = estimate(std,X)/sqrt(N)
                    err_avg2Im[i] = std(X)/sqrt(N/nn)
                end
            end
            
            @. X = phi_p_Re_1 * phi_p_Re_i - phi_p_Im_1 * phi_p_Im_i
            corr0tRe[i] = mean(X)
            if bstrap
                err_corr0tRe[i] = stderror(bootstrap(std, X, S))[1]
            else
                #err_corr0tRe[i] = estimate(std,X)/sqrt(N)
                err_corr0tRe[i] = std(X)/sqrt(N/nn)
            end

            @. X = phi_p_Re_i * phi_p_Im_1 + phi_p_Re_1 * phi_p_Im_i
            corr0tIm[i] = mean(X)
            if bstrap
                err_corr0tIm[i] = stderror(bootstrap(std, X, S))[1]
            else
                #err_corr0tIm[i] = estimate(std,X)/sqrt(N)
                err_corr0tIm[i] = std(X)/sqrt(N/nn)
            end
        end
        return avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm

    elseif sol isa SciMLBase.EnsembleSolution

        avgRe = zeros(T,length(sol),t_steps)
        avgIm = zeros(T,length(sol),t_steps)
        avg2Re = zeros(T,length(sol),t_steps)
        avg2Im = zeros(T,length(sol),t_steps)
        corr0tRe = zeros(T,length(sol),t_steps)
        corr0tIm = zeros(T,length(sol),t_steps)

        X = Array(sol)

        for i in eachindex(sol)
            _u = @view X[:,:,i]
            _uRe = @view _u[1:t_steps*n_steps^D,:]
            _uIm = @view _u[t_steps*n_steps^D + 1:end,:]
            
            _uu = zeros(2t_steps,size(_u)[2])
            for i in 1:n_steps
                _uu[1:t_steps,:] .= _uRe[(i-1)*t_steps .+ (1:t_steps),:]
                _uu[t_steps+1:end,:] .= _uIm[(i-1)*t_steps .+ (1:t_steps),:]
            end

            phi_p_Re = (@view _uu[1:t_steps,:]) / n_steps^D #mean(_uu[1:t_steps,:,:,:],dims=(2,3))[:,1,1,:]
            phi_p_Im = (@view _uu[t_steps+1:end,:]) / n_steps^D #mean(_uu[t_steps+1:end,:,:,:],dims=(2,3))[:,1,1,:]

            avgRe[i,:] .= mean(phi_p_Re,dims=2)[:,1]
            avgIm[i,:] .= mean(phi_p_Im,dims=2)[:,1]

            avg2Re[i,:] .= mean(phi_p_Re.^2 .- phi_p_Im.^2,dims=2)[:,1]
            avg2Im[i,:] .= 2*mean(phi_p_Re .* phi_p_Im,dims=2)[:,1]

            corr0tRe[i,:] .= mean(phi_p_Re[1:1,:] .* phi_p_Re .- phi_p_Im[1:1,:] .* phi_p_Im,dims=2)[:,1]
            corr0tIm[i,:] .= mean(phi_p_Re .* phi_p_Im[1:1,:] .+ phi_p_Re[1:1,:] .* phi_p_Im,dims=2)[:,1]
        end

        return avgRe, avgIm, avg2Re, avg2Im, corr0tRe, corr0tIm
    end
end


"""
    Calculate the observable means over the different trajectories with the corresponding error
"""
function calc_meanObs(obs)
    avgRe, avgIm, avg2Re, avg2Im, corr0tRe, corr0tIm = obs

    NTr = length(avgRe) 

    d=1
    return mean(avgRe,dims=d)[1,:], (std(avgRe,dims=d)/sqrt(NTr))[1,:], 
        mean(avgIm,dims=d)[1,:], (std(avgIm,dims=d)/sqrt(NTr))[1,:],
        mean(avg2Re,dims=d)[1,:], (std(avg2Re,dims=d)/sqrt(NTr))[1,:], 
        mean(avg2Im,dims=d)[1,:], (std(avg2Im,dims=d)/sqrt(NTr))[1,:],
        mean(corr0tRe,dims=d)[1,:], (std(corr0tRe,dims=d)/sqrt(NTr))[1,:], 
        mean(corr0tIm,dims=d)[1,:], (std(corr0tIm,dims=d)/sqrt(NTr))[1,:]

end


"""
    Calculate the loss based on the symmetry prior knowledge
"""
function calcSymLoss(sol,KP::KernelProblem{ScalarField{D}}; obs = nothing) where {D}

    if isnothing(obs)
        obs = calc_obs(KP,sol)
    end
    
    if sol isa SciMLBase.RODESolution || length(sol) == 1
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs
    elseif sol isa SciMLBase.EnsembleSolution
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(obs)
    end


    normalized_err = maximum(err_avg2Re ./ abs(avg2Re[1]))

    if "phi2Re" in keys(KP.y)
        return normalized_err * sum(abs2, [ (KP.y["phiRe"] .- avgRe) ./ err_avgRe;
                        (KP.y["phiIm"] .- avgIm) ./  err_avgIm;
                        (real(KP.y["phi2Re"]) .- avg2Re) ./  err_avg2Re; 
                        (imag(KP.y["phi2Im"]) .- avg2Im) ./ err_avg2Im;
                ])
    else
        return normalized_err * sum(abs2, [ (KP.y["phiRe"] .- avgRe) ./ err_avgRe;
                        (KP.y["phiIm"] .- avgIm) ./  err_avgIm;
                ])
    end
end





"""
    Calculate loss gradient
"""
function getdH_imx(sol,KP::KernelProblem{ScalarField{D}}) where {D}
    dKs = [zeros(size(KP.kernel.H)) for i in 1:length(sol)]
    for (i,tr) in enumerate(sol)
        dKs[i] = Zygote.gradient(H -> calcIMXLoss(tr,KP, H=H), KP.kernel.H)[1]
    end

    return reshape(mean(dKs),size(KP.kernel.H))
end