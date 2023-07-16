
"""
    Get the drift and noise term to be used in the simulation
"""
function get_ab(model::ScalarField{1},kernel::MatrixKernel{T}) where {T <: Real}
    
    @unpack m, λ, contour, n_steps, as = model
    @unpack a, t_steps = contour
    @unpack H = kernel

    KRe, KIm = getK(kernel)

    gm1=vcat([t_steps],1:t_steps-1)
    gp1=vcat(2:t_steps,[1])

    gsm1=vcat([n_steps],1:n_steps-1)
    gsp1=vcat(2:n_steps,[1])
    
    # if CUDA.functional() && USE_GPU
    #     KRe = CuArray(Float32.(KRe))
    #     KIm = CuArray(Float32.(KIm))
    # elseif Metal.functional() && USE_GPU
    #     KRe = MtlArray(Float32.(KRe))
    #     KIm = MtlArray(Float32.(KIm))

        
    #     a = MtlArray(Complex{Float32}.(a))
    #     a_m1 = MtlArray(Complex{Float32}.(a[gm1]))
        
    # else
        a_m1 = a[gm1]
    # end

    # gm1=vcat([t_steps],1:t_steps-1)
    # gp1=vcat(2:t_steps,[1])

    # gsm1=vcat([n_steps],1:n_steps-1)
    # gsp1=vcat(2:n_steps,[1])

    #a_2a2_Re = hcat([real(a) ./ (2 * as^2)  for i in 1:n_steps]...)
    #a_2a2_Im = hcat([imag(a) ./ (2 * as^2)  for i in 1:n_steps]...)
    
    #a_m1 = a[gm1]
    #atm1_2a2_Re = hcat([real(a_m1) ./ (2 * as^2)  for i in 1:n_steps]...)
    #atm1_2a2_Im = hcat([imag(a_m1) ./ (2 * as^2)  for i in 1:n_steps]...)
    as_prefac_re = 1 / (as^2) # (atm1_2a2_Re + a_2a2_Re)
    as_prefac_im = 1/ (as^2) #(atm1_2a2_Im + a_2a2_Im)

    one_over_a = a.^(-1)
    one_over_a_Re = hcat([real(one_over_a) for i in 1:n_steps]...)
    one_over_a_Im = hcat([imag(one_over_a) for i in 1:n_steps]...)
    one_over_a_m1 = a_m1.^(-1)
    one_over_a_m1_Re = hcat([real(one_over_a_m1) for i in 1:n_steps]...)
    one_over_a_m1_Im = hcat([imag(one_over_a_m1) for i in 1:n_steps]...)
    
    V_pre_fac = (a + a_m1)/2
    V_pre_fac_Re = hcat([real(V_pre_fac) for i in 1:n_steps]...)
    V_pre_fac_Im = hcat([imag(V_pre_fac) for i in 1:n_steps]...)

    pre_fac = (1. / abs(a[1]))
    #im_pre_fac = im*pre_fac

    function a_func!(du,u,p,t)

        _uRe = @view u[1:t_steps*n_steps]
        _uIm = @view u[t_steps*n_steps + 1:end]

        uRe = reshape(_uRe,t_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps)
        #_u = u

        tmp = get_tmp(p, u)
        ARe = @view tmp[1:t_steps,:]
        AIm = @view tmp[t_steps .+ (1:t_steps),:]
        VRe = @view tmp[2t_steps .+ (1:t_steps),:]
        VIm = @view tmp[3t_steps .+ (1:t_steps),:]

        #uRe = @view _u[1:t_steps,:]
        uRetm1 = @view uRe[gm1,:]
        uRetp1 = @view uRe[gp1,:]
        uResm1 = @view uRe[1:t_steps,gsm1]
        uResp1 = @view uRe[1:t_steps,gsp1]
        
        #uIm = @view _u[t_steps+1:end,:]
        uImtm1 = @view uIm[gm1,:]
        uImtp1 = @view uIm[gp1,:]
        uImsm1 = @view uIm[1:t_steps,gsm1]
        uImsp1 = @view uIm[1:t_steps,gsp1]
        
        @. VRe = m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
        @. VIm = m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)

        @. ARe = - as * pre_fac * (
            (uRe - uRetm1)*one_over_a_m1_Im 
           + (uIm - uImtm1)*one_over_a_m1_Re  
           + (uRe - uRetp1)*one_over_a_Im 
           + (uIm - uImtp1)*one_over_a_Re

           - V_pre_fac_Im * ( (2uRe - uResm1 - uResp1)*as_prefac_im + VRe)
           - V_pre_fac_Re * ( (2uIm - uImsm1 - uImsp1)*as_prefac_re + VIm)
        )

        @. AIm =  as * pre_fac * (
            (uRe - uRetm1)*one_over_a_m1_Re 
           - (uIm - uImtm1)*one_over_a_m1_Im  
           + (uRe - uRetp1)*one_over_a_Re 
           - (uIm - uImtp1)*one_over_a_Im

           - V_pre_fac_Re * ( (2uRe - uResm1 - uResp1)*as_prefac_re + VRe)
           + V_pre_fac_Im * ( (2uIm - uImsm1 - uImsp1)*as_prefac_im + VIm)
        )

        

        # if CUDA.functional() && USE_GPU
        #     _ARe = CuArray(vec(ARe))
        #     _AIm = CuArray(vec(AIm))
        #     du[1:t_steps*n_steps,:] .= Array(KRe*_ARe - KIm*_AIm)
        #     du[t_steps*n_steps + 1:end,:] .= Array(KIm*_ARe + KRe*_AIm)
        # elseif Metal.functional() && USE_GPU
        #     _ARe = MtlArray(vec(ARe))
        #     _AIm = MtlArray(vec(ARe))
        #     du[1:t_steps*n_steps,:] .= Array(KRe*_ARe - KIm*_AIm)
        #     du[t_steps*n_steps + 1:end,:] .= Array(KIm*_ARe + KRe*_AIm)
        # else
            _ARe = vec(ARe)
            _AIm = vec(AIm)
            du[1:t_steps*n_steps,:] .= KRe*_ARe - KIm*_AIm
            du[t_steps*n_steps + 1:end,:] .= KIm*_ARe + KRe*_AIm
        # end
        


        nothing
    end
    
    #@warn "Using Diagonal noise, without a kernel"
    #sqrt2pre_fac = sqrt(2 * pre_fac)
    sqrt2pre_fac_sqrtK = sqrt(2 * pre_fac) * H
    
    function b_func!(du,u,p,t)
        
        @. du = sqrt2pre_fac_sqrtK
        nothing
    end 

    return a_func!, b_func!
end



"""
    Calculate the observables for each of the trajectories
"""
function calc_obs(KP::KernelProblem{ScalarField{D}},sol;onlyCorr=false) where {D}
    t_steps = KP.model.contour.t_steps
    n_steps = KP.model.n_steps

    T = eltype(sol[1]) #eltype( getKernelParams(KP.kernel) )
    
    if length(sol) == 1
        sol = sol[1]
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

        _u = Array(sol)
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
    Caluclate the true loss
"""
#=function calcTrueLoss(sol,KP::KernelProblem{AHO}; obs_calc = nothing)
    
    if isnothing(obs_calc)
        obs = calc_obs(KP,sol)
        corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol)) 
    else
        corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs_calc
    end

    normalized_err = maximum(err_avg2Re ./ real(KP.y["x2"]))

    return normalized_err * sum([(real(KP.y["corr0t"]) .- corr0tRe) ./ err_corr0tRe; 
                                (imag(KP.y["corr0t"]) .- corr0tIm) ./ err_corr0tIm])
end=#

#=
"""
    Calculate the loss based on the symmetry prior knowledge
"""
function calcSymLoss(sol,KP::KernelProblem{AHO11}; obs_calc = nothing)

    
    #βsteps = KP.model.contour.EucledianSteps

    if isnothing(obs_calc)
        obs = calc_obs(KP,sol)
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol)) 
    else
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs_calc
    end


    normalized_err = maximum(err_avg2Re ./ abs(avg2Re[1]))

    return normalized_err * sum(abs2, [ (0. .- avgRe) ./ err_avgRe;
                    (0 .- avgIm) ./  err_avgIm;
                    (real(KP.y["x2"]) .- avg2Re) ./  err_avg2Re; 
                    (imag(KP.y["x2"]) .- avg2Im) ./ err_avg2Im;
                    #(real(KP.y["corr0t"][end-βsteps]) .- corr0tRe)[end-βsteps] ./ err_corr0tRe[end-βsteps]; 
                    #(imag(KP.y["corr0t"][end-βsteps]) .- corr0tIm)[end-βsteps] ./ err_corr0tIm[end-βsteps]
            ])
end


"""
    Calculate the drift loss used to update approximate the gradient
"""
function calcDriftLoss(sol,KP::KernelProblem{AHO11,T};p=getKernelParams(KP.kernel)) where {T <: ConstantKernel}

    @unpack m, λ, contour, n_steps, as = KP.model
    @unpack a, t_steps = contour
    @unpack sqrtK,K = KP.kernel

    
    
    ξ = 1.5
    

    gtm1=vcat([t_steps],1:t_steps-1)
    gtp1=vcat(2:t_steps,[1])

    gsm1=vcat([n_steps],1:n_steps-1)
    gsp1=vcat(2:n_steps,[1])

    
    a_m1 = a[gtm1]
    as_prefac = 1 / (as^2) # (atm1_2a2_Re + a_2a2_Re)
    #as_prefac_im = 1/ (as^2)

    
    pre_fac = (as / abs(a[1]))
    
    KRe,KIm = K([],p)
    KC = KRe .+ im*KIm
    
    im_pre_fac_KC = KC*im*pre_fac

    dt = 1e-5
    κ = im*1e-2

    g(u) = begin

        _uRe = @view u[1:t_steps*n_steps]
        _uIm = @view u[t_steps*n_steps + 1:end]

        uRe = reshape(_uRe,t_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps)

        #_uu = reshape(u,2t_steps,n_steps)
        _x = uRe + im * uIm
        #_x = (@view u[1:t_steps*]) + im * (@view u[t_steps*n_steps+1:end])
        for i in 1:5
            _A_tmp =    @. ((_x - _x[gtm1,:]) / (a_m1 .- κ) + (_x - _x[gtp1,:]) / (a .- κ)  - 
            (a + a_m1)/2 * ( (2*_x - _x[:,gsm1] - _x[:,gsp1]) * as_prefac 
                                +  (m * _x + (λ/6) * _x^3)))

            _A = im_pre_fac_KC * vec(_A_tmp)

            _x += reshape(_A * dt,t_steps,n_steps)
        end

        
        #return _x
        return sum(imag(_x).^2)
        #return abs(real(adjoint(_A) * (-vec(_x))) - norm(_A) * norm(vec(_x)))^ξ        
        #return norm(imag(_A))^ξ        
        #return abs.(real(conj.(_A) .* (-vec(_x))) - norm.(_A) .* norm.(vec(_x))).^ξ     
        #return abs(real(adjoint(imag(_A)) * (-vec(imag(_x)))) - norm(imag(_A)) * norm(vec(imag(_x))))^ξ        

    end


    #=XX = [g(u) for u in eachrow(sol')]

    xRe = sum( abs2.(StatsBase.mean([real(_x) for _x in XX]) ) )
    xIm = sum( abs2.(StatsBase.mean([imag(_x) for _x in XX]) ) )

    x2Re = sum( abs2.(StatsBase.mean([real(_x).^2 .- imag(_x).^2 for _x in XX]) .- real(KP.y["x2"])) )
    x2Im = sum( abs2.(StatsBase.mean([2 .* real(_x) .* imag(_x) for _x in XX]) .- imag(KP.y["x2"])) )

    imx = sum( abs.(StatsBase.mean([imag(_x).^2 for _x in XX])) )

    return xRe + xIm + x2Re + x2Im + imx=#

    return sum(
        mean(g(u) for u in eachrow(sol'))
    #        mean(g(view(sol,:,ii)) for ii in 1:size(sol)[2])
        )

end
=#