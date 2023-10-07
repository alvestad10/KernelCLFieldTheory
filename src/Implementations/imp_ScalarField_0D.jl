
"""
    Get the drift and noise term to be used in the simulation
"""
function get_ab(model::ScalarField{0},kernel::T) where {T <: ConstantKernel}
    
    @unpack m, λ, contour = model
    @unpack a, t_steps = contour
    @unpack H = kernel

    KRe, KIm = getK(kernel)

    gm1=vcat([t_steps],1:t_steps-1)
    gp1=vcat(2:t_steps,[1])

    a_m1 = a[gm1]

    one_over_a = a.^(-1)
    one_over_a_Re = real(one_over_a) #hcat([real(one_over_a) for i in 1:n_steps]...)
    one_over_a_Im = imag(one_over_a) #hcat([imag(one_over_a) for i in 1:n_steps]...)
    one_over_a_m1 = a_m1.^(-1)
    one_over_a_m1_Re = real(one_over_a_m1) #hcat([real(one_over_a_m1) for i in 1:n_steps]...)
    one_over_a_m1_Im = imag(one_over_a_m1) #hcat([imag(one_over_a_m1) for i in 1:n_steps]...)
    
    V_pre_fac = (a + a_m1)/2
    V_pre_fac_Re = real(V_pre_fac) #hcat([real(V_pre_fac) for i in 1:n_steps]...)
    V_pre_fac_Im = imag(V_pre_fac) #hcat([imag(V_pre_fac) for i in 1:n_steps]...)

    pre_fac = (1. / abs(a[1]))
    #im_pre_fac = im*pre_fac

    isIdentity = all(diag(H) .- 1. .== 0.0)

    function a_func!(du,u,p,t)

        uRe = @view u[1:t_steps]
        uIm = @view u[t_steps + 1:end]

        #_u = u

        tmp = get_tmp(p, u)
        ARe = @view tmp[1:t_steps]
        AIm = @view tmp[t_steps .+ (1:t_steps)]
        VRe = @view tmp[2t_steps .+ (1:t_steps)]
        VIm = @view tmp[3t_steps .+ (1:t_steps)]
        
        if !isIdentity
            KReARe = vec(@view tmp[4t_steps .+ (1:t_steps)])
            KImAIm = vec(@view tmp[5t_steps .+ (1:t_steps)])
            KReAIm = vec(@view tmp[6t_steps .+ (1:t_steps)])
            KImARe = vec(@view tmp[7t_steps .+ (1:t_steps)])
        end

        #uRe = @view _u[1:t_steps,:]
        uRetm1 = @view uRe[gm1]
        uRetp1 = @view uRe[gp1]
        
        #uIm = @view _u[t_steps+1:end,:]
        uImtm1 = @view uIm[gm1]
        uImtp1 = @view uIm[gp1]
        
        @. VRe = m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
        @. VIm = m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)
 

        @inbounds @fastmath for i in 1:t_steps
            ARe[i] = - pre_fac * (
                    (uRe[i] - uRetm1[i])*one_over_a_m1_Im[i]
                + (uIm[i] - uImtm1[i])*one_over_a_m1_Re[i]  
                + (uRe[i] - uRetp1[i])*one_over_a_Im[i] 
                + (uIm[i] - uImtp1[i])*one_over_a_Re[i]

                - V_pre_fac_Im[i] * VRe[i]
                - V_pre_fac_Re[i] * VIm[i]
                )
            
            AIm[i] = pre_fac * (
                    (uRe[i] - uRetm1[i])*one_over_a_m1_Re[i] 
                - (uIm[i] - uImtm1[i])*one_over_a_m1_Im[i]  
                + (uRe[i] - uRetp1[i])*one_over_a_Re[i] 
                - (uIm[i] - uImtp1[i])*one_over_a_Im[i]
        
                - V_pre_fac_Re[i] * VRe[i]
                + V_pre_fac_Im[i] * VIm[i]
                )
        end

        if isIdentity
            du[1:t_steps,:] .= vec(ARe)
            du[t_steps + 1:end,:] .= vec(AIm)
        else
            _ARe = vec(ARe)
            _AIm = vec(AIm)
            KReARe .= KRe*_ARe
            KImAIm .= KIm*_AIm
            KReAIm .= KRe*_AIm
            KImARe .= KIm*_ARe
            @. du[1:t_steps,:] = KReARe - KImAIm
            @. du[t_steps + 1:end,:] = KImARe + KReAIm
        end
        nothing
    end
    
    
    sqrt2pre_fac = sqrt(2 * pre_fac)
    sqrt2pre_fac_sqrtK = sqrt(2 * pre_fac) * H
    
    function b_func!(du,u,p,t)
        if isIdentity
            @. du[1:div(end,2)] = sqrt2pre_fac
        else
            @. du = sqrt2pre_fac_sqrtK
        end
        nothing
    end 

    return a_func!, b_func!
end




# """
#     Caluclate the true loss
# """
# function calcTrueLoss(sol,KP::KernelProblem{AHO}; obs_calc = nothing)
    
#     if isnothing(obs_calc)
#         obs = calc_obs(KP,sol)
#         corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(obs) 
#     else
#         corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs_calc
#     end

#     normalized_err = maximum(err_avg2Re ./ real(KP.y["x2"]))

#     return normalized_err * sum([(real(KP.y["corr0t"]) .- corr0tRe) ./ err_corr0tRe; 
#                                 (imag(KP.y["corr0t"]) .- corr0tIm) ./ err_corr0tIm])
# end







"""
    Calculate the drift loss used to update approximate the gradient
"""
function calcIMXLoss(sol_tr,KP::KernelProblem{ScalarField{0}}; H = KP.kernel.H)

    @unpack m, λ, contour = KP.model
    @unpack a, t_steps = contour
    

    gtm1=vcat([t_steps],1:t_steps-1)
    gtp1=vcat(2:t_steps,[1])

    dt = 1e-5
    κ = im*1e-2
    
    a_m1 = a[gtm1]

    one_over_a = 1 ./ (a .- κ)
    one_over_am1 = 1 ./ (a_m1 .- κ)
    
    pre_fac = (1 / abs(a[1]))
    
    KRe,KIm = getK(H)
    KC = KRe .+ im*KIm

    a_ap1_fac = (a .+ a_m1)/2
    
    im_pre_fac_KC = KC*im*pre_fac

    

    g(u) = begin

        uRe = @view u[1:t_steps]
        uIm = @view u[t_steps + 1:end]
        
        
        _x = uRe + im * uIm
        _xtm1 = @view _x[gtm1]
        _xtp1 = @view _x[gtp1]

        for i in 1:5

            @tullio _A_tmp[i] := ((_x[i] - _xtm1[i]) * one_over_am1[i]  + (_x[i] - _xtp1[i]) * one_over_a[i]  - 
                    a_ap1_fac[i] * (m * _x[i] + (λ/6) * _x[i]^3))

            _A = im_pre_fac_KC * vec(_A_tmp)

            _x += _A * dt
        end

        
        return _x
        #return sum(imag(_x).^2) # + sum(real(_x).^2)

    end

    XX = [g(u) for u in eachrow(sol_tr')]

    # xRe = sum( abs2.(StatsBase.mean([real(X) for X in XX]) ) )
    # xIm = sum( abs2.(StatsBase.mean([imag(X) for X in XX]) ) )
    
    # x2Re = sum( abs2.(StatsBase.mean([real(X).^2 .- imag(X).^2 for X in XX]) .- KP.y["phi2Re"]) )
    # x2Im = sum( abs2.(StatsBase.mean([2 .* real(X) .* imag(X) for X in XX]) .- KP.y["phi2Im"]) )

    #imx = sum([maximum(imag(_x).^2) for _x in XX])
    #rex = sum([maximum(real(_x).^2) for _x in XX])

    imx = sum([sum(imag(_x).^2) for _x in XX])
    #rex = sum([sum(real(_x).^2) for _x in XX])


    #return xRe + xIm + 5*x2Re + x2Im + imx + rex
    return imx #+ rex
    # return sum(
    #     mean(g(u) for u in eachrow(sol_tr'))
    #     )

end