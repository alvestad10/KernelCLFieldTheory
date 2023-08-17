
"""
    Get the drift and noise term to be used in the simulation
"""
function get_ab(model::ScalarField{2},kernel::MatrixKernel{T}) where {T <: Real}
    
    @unpack m, λ, contour, n_steps, as = model
    @unpack a, t_steps = contour
    @unpack H = kernel

    KRe, KIm = getK(kernel)

    CUDA.allowscalar(false)
    use_GPU = true
    if use_GPU
        KRe = CuArray(KRe)
        KIm = CuArray(KIm)
    end

    gm1=vcat([t_steps],1:t_steps-1)
    gp1=vcat(2:t_steps,[1])

    gsm1=vcat([n_steps],1:n_steps-1)
    gsp1=vcat(2:n_steps,[1])
    
    a_m1 = a[gm1]
    as_prefac_re = 1 / (as^2)
    as_prefac_im = 1/ (as^2) 

    one_over_a = a.^(-1)
    one_over_a_Re = real(one_over_a) #hcat([real(one_over_a) for i in 1:n_steps]...)
    one_over_a_Im = imag(one_over_a) #hcat([imag(one_over_a) for i in 1:n_steps]...)
    one_over_a_m1 = a_m1.^(-1)
    one_over_a_m1_Re = real(one_over_a_m1) #hcat([real(one_over_a_m1) for i in 1:n_steps]...)
    one_over_a_m1_Im = imag(one_over_a_m1) #hcat([imag(one_over_a_m1) for i in 1:n_steps]...)
    
    V_pre_fac = (a + a_m1)/2
    V_pre_fac_Re = real(V_pre_fac) #hcat([real(V_pre_fac) for i in 1:n_steps]...)
    V_pre_fac_Im = imag(V_pre_fac) #hcat([imag(V_pre_fac) for i in 1:n_steps]...)

    pre_fac = (1. / (abs(a[1])))
    #im_pre_fac = im*pre_fac

    isIdentity = all(diag(H) .- 1. .== 0.0)

    function a_func!(du,u,p,t)

        _uRe = @view u[1:t_steps*n_steps^2]
        _uIm = @view u[t_steps*n_steps^2 + 1:end]

        uRe = reshape(_uRe,t_steps,n_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps,n_steps)
        #_u = u

        tmp = get_tmp(p, u)
        ARe = @view tmp[1:t_steps,:,:]
        AIm = @view tmp[t_steps .+ (1:t_steps),:,:]
        VRe = @view tmp[2t_steps .+ (1:t_steps),:,:]
        VIm = @view tmp[3t_steps .+ (1:t_steps),:,:]
        
        if !isIdentity
            KReARe = vec(@view tmp[4t_steps .+ (1:t_steps),:,:])
            KImAIm = vec(@view tmp[5t_steps .+ (1:t_steps),:,:])
            KReAIm = vec(@view tmp[6t_steps .+ (1:t_steps),:,:])
            KImARe = vec(@view tmp[7t_steps .+ (1:t_steps),:,:])
        end

        #uRe = @view _u[1:t_steps,:]
        uRetm1 = @view uRe[gm1,:,:]
        uRetp1 = @view uRe[gp1,:,:]
        uRes1m1 = @view uRe[1:t_steps,gsm1,:]
        uRes1p1 = @view uRe[1:t_steps,gsp1,:]
        uRes2m1 = @view uRe[1:t_steps,:,gsm1]
        uRes2p1 = @view uRe[1:t_steps,:,gsp1]
        
        #uIm = @view _u[t_steps+1:end,:]
        uImtm1 = @view uIm[gm1,:,:]
        uImtp1 = @view uIm[gp1,:,:]
        uIms1m1 = @view uIm[1:t_steps,gsm1,:]
        uIms1p1 = @view uIm[1:t_steps,gsp1,:]
        uIms2m1 = @view uIm[1:t_steps,:,gsm1]
        uIms2p1 = @view uIm[1:t_steps,:,gsp1]
        
        @. VRe = m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
        @. VIm = m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)
 

        @inbounds @fastmath for I in CartesianIndices((t_steps,n_steps,n_steps))
            i,j,k = Tuple(I)
            ARe[i,j,k] = - as^2 * pre_fac * (
                    (uRe[i,j,k] - uRetm1[i,j,k])*one_over_a_m1_Im[i]
                + (uIm[i,j,k] - uImtm1[i,j,k])*one_over_a_m1_Re[i]  
                + (uRe[i,j,k] - uRetp1[i,j,k])*one_over_a_Im[i] 
                + (uIm[i,j,k] - uImtp1[i,j,k])*one_over_a_Re[i]

                - V_pre_fac_Im[i] * ( (2uRe[i,j,k] - uRes1m1[i,j,k] - uRes1p1[i,j,k])*as_prefac_im + 
                                        (2uRe[i,j,k] - uRes2m1[i,j,k] - uRes2p1[i,j,k])*as_prefac_im +
                                        VRe[i,j,k])
                - V_pre_fac_Re[i] * ( (2uIm[i,j,k] - uIms1m1[i,j,k] - uIms1p1[i,j,k])*as_prefac_re + 
                                        (2uIm[i,j,k] - uIms2m1[i,j,k] - uIms2p1[i,j,k])*as_prefac_re +
                                        VIm[i,j,k])
                )
            
            AIm[i,j,k] = as^2 * pre_fac * (
                    (uRe[i,j,k] - uRetm1[i,j,k])*one_over_a_m1_Re[i] 
                - (uIm[i,j,k] - uImtm1[i,j,k])*one_over_a_m1_Im[i]  
                + (uRe[i,j,k] - uRetp1[i,j,k])*one_over_a_Re[i] 
                - (uIm[i,j,k] - uImtp1[i,j,k])*one_over_a_Im[i]
        
                - V_pre_fac_Re[i] * ( (2uRe[i,j,k] - uRes1m1[i,j,k] - uRes1p1[i,j,k])*as_prefac_re + 
                                        (2uRe[i,j,k] - uRes2m1[i,j,k] - uRes2p1[i,j,k])*as_prefac_re +    
                                        VRe[i,j,k])
                + V_pre_fac_Im[i] * ( (2uIm[i,j,k] - uIms1m1[i,j,k] - uIms1p1[i,j,k])*as_prefac_im +
                                        (2uIm[i,j,k] - uIms2m1[i,j,k] - uIms2p1[i,j,k])*as_prefac_im + 
                                    VIm[i,j,k])
                )
                #end
            #end
        end

        if isIdentity
            du[1:t_steps*n_steps^2,:] .= vec(ARe)
            du[t_steps*n_steps^2 + 1:end,:] .= vec(AIm)
        elseif use_GPU
            _ARe = CuArray(vec(ARe))
            _AIm = CuArray(vec(AIm))
            copyto!(du[1:t_steps*n_steps^2,:],KRe*_ARe - KIm*_AIm)
            copyto!(du[t_steps*n_steps^2 + 1:end,:],KIm*_ARe + KRe*_AIm)
        else
            _ARe = vec(ARe)
            _AIm = vec(AIm)
            KReARe .= KRe*_ARe
            KImAIm .= KIm*_AIm
            KReAIm .= KRe*_AIm
            KImARe .= KIm*_ARe
            @. du[1:t_steps*n_steps^2,:] = KReARe - KImAIm
            @. du[t_steps*n_steps^2 + 1:end,:] = KImARe + KReAIm
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
function calcIMXLoss(sol_tr,KP::KernelProblem{ScalarField{2}}; H = KP.kernel.H)

    @unpack m, λ, contour, n_steps, as = KP.model
    @unpack a, t_steps = contour
    

    gtm1=vcat([t_steps],1:t_steps-1)
    gtp1=vcat(2:t_steps,[1])

    gsm1=vcat([n_steps],1:n_steps-1)
    gsp1=vcat(2:n_steps,[1])

    
    a_m1 = a[gtm1]
    as_prefac = 1 / (as^2) 
    
    pre_fac = (as^2 / abs(a[1]))
    
    KRe,KIm = getK(H)
    KC = KRe .+ im*KIm
    
    im_pre_fac_KC = KC*im*pre_fac

    dt = 1e-5
    κ = im*1e-2

    g(u) = begin

        _uRe = @view u[1:t_steps*n_steps^2]
        _uIm = @view u[t_steps*n_steps^2 + 1:end]
        
        uRe = reshape(_uRe,t_steps,n_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps,n_steps)
        
        _x = uRe + im * uIm
        _xtm1 = @view _x[gtm1,:,:]
        _xtp1 = @view _x[gtp1,:,:]
        _xs1m1 = @view _x[:,gsm1,:]
        _xs1p1 = @view _x[:,gsp1,:]
        _xs2m1 = @view _x[:,:,gsm1]
        _xs2p1 = @view _x[:,:,gsp1]

        for i in 1:5

            @tullio _A_tmp[i,j,k] := ((_x[i,j,k] - _xtm1[i,j,k]) / (a_m1[i] - κ) + (_x[i,j,k] - _xtp1[i,j,k]) / (a[i] - κ)  - 
                            (a[i] + a_m1[i])/2 * ( (2*_x[i,j,k] - _xs1m1[i,j,k] - _xs1p1[i,j,k]) * as_prefac +
                                                    (2*_x[i,j,k] - _xs2m1[i,j,k] - _xs2p1[i,j,k]) * as_prefac  
                                +  (m * _x[i,j,k] + (λ/6) * _x[i,j,k]^3)))

            _A = im_pre_fac_KC * vec(_A_tmp)

            _x += reshape(_A * dt,t_steps,n_steps,n_steps)
        end

        
        return _x
        #return sum(imag(_x).^2) # + sum(real(_x).^2)

    end

    XX = [g(u) for u in eachrow(sol_tr')]
    XP0 = [mean(X,dims=(2,3)) for X in XX]

    xRe = sum( abs2.(StatsBase.mean([real(X) for X in XP0]) .- 0 )  )
    xIm = sum( abs2.(StatsBase.mean([imag(X) for X in XP0]) .- 0 )  )

    x2Re = sum( abs2.(StatsBase.mean([real(X).^2 .- imag(X).^2 for X in XP0]) .- KP.y["phi2Re"]) )
    x2Im = sum( abs2.(StatsBase.mean([2 .* real(X) .* imag(X) for X in XP0]) .- KP.y["phi2Im"]) )

    imx = sum( abs.(StatsBase.sum([imag(_x).^2 for _x in XX])) )
    rex = sum( abs.(StatsBase.sum([real(_x).^2 for _x in XX])) )

    return xRe + xIm + x2Re + x2Im + 10*imx + rex
    
    #return  imx #+ x2Re 
    
    # return sum(
    #     mean(g(u) for u in eachrow(sol_tr'))
    #     )

end