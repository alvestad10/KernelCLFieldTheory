
"""
    Get the drift and noise term to be used in the simulation
"""
function get_ab(model::ScalarField{3},kernel::MatrixKernel{T}) where {T <: Real}
    
    @unpack m, λ, contour, n_steps, as = model
    @unpack a, t_steps = contour
    @unpack H = kernel

    KRe, KIm = getK(kernel)

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

    pre_fac = (1. / abs(a[1]))
    #im_pre_fac = im*pre_fac

    isIdentity = all(diag(H) .- 1. .== 0.0)

    function a_func!(du,u,p,t)

        _uRe = @view u[1:t_steps*n_steps^3]
        _uIm = @view u[t_steps*n_steps^3 + 1:end]

        uRe = reshape(_uRe,t_steps,n_steps,n_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps,n_steps,n_steps)
        #_u = u

        tmp = get_tmp(p, u)
        ARe = @view tmp[1:t_steps,:,:,:]
        AIm = @view tmp[t_steps .+ (1:t_steps),:,:,:]
        VRe = @view tmp[2t_steps .+ (1:t_steps),:,:,:]
        VIm = @view tmp[3t_steps .+ (1:t_steps),:,:,:]
        
        if !isIdentity
            KReARe = vec(@view tmp[4t_steps .+ (1:t_steps),:,:,:])
            KImAIm = vec(@view tmp[5t_steps .+ (1:t_steps),:,:,:])
            KReAIm = vec(@view tmp[6t_steps .+ (1:t_steps),:,:,:])
            KImARe = vec(@view tmp[7t_steps .+ (1:t_steps),:,:,:])
        end

        #uRe = @view _u[1:t_steps,:]
        uRetm1 = @view uRe[gm1,:,:,:]
        uRetp1 = @view uRe[gp1,:,:,:]
        uRes1m1 = @view uRe[1:t_steps,gsm1,:,:]
        uRes1p1 = @view uRe[1:t_steps,gsp1,:,:]
        uRes2m1 = @view uRe[1:t_steps,:,gsm1,:]
        uRes2p1 = @view uRe[1:t_steps,:,gsp1,:]
        uRes3m1 = @view uRe[1:t_steps,:,:,gsm1]
        uRes3p1 = @view uRe[1:t_steps,:,:,gsp1]
        
        #uIm = @view _u[t_steps+1:end,:]
        uImtm1 = @view uIm[gm1,:,:,:]
        uImtp1 = @view uIm[gp1,:,:,:]
        uIms1m1 = @view uIm[1:t_steps,gsm1,:,:]
        uIms1p1 = @view uIm[1:t_steps,gsp1,:,:]
        uIms2m1 = @view uIm[1:t_steps,:,gsm1,:]
        uIms2p1 = @view uIm[1:t_steps,:,gsp1,:]
        uIms3m1 = @view uIm[1:t_steps,:,:,gsm1]
        uIms3p1 = @view uIm[1:t_steps,:,:,gsp1]
        
        @. VRe = m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
        @. VIm = m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)
 

        @inbounds for I in CartesianIndices((t_steps,n_steps,n_steps,n_steps))
            i,j,k,l = Tuple(I)
            ARe[i,j,k,l] = - as^3 * pre_fac * (
                    (uRe[i,j,k,l] - uRetm1[i,j,k,l])*one_over_a_m1_Im[i]
                + (uIm[i,j,k,l] - uImtm1[i,j,k,l])*one_over_a_m1_Re[i]  
                + (uRe[i,j,k,l] - uRetp1[i,j,k,l])*one_over_a_Im[i] 
                + (uIm[i,j,k,l] - uImtp1[i,j,k,l])*one_over_a_Re[i]

                - V_pre_fac_Im[i] * ( (2uRe[i,j,k,l] - uRes1m1[i,j,k,l] - uRes1p1[i,j,k,l])*as_prefac_im + 
                                        (2uRe[i,j,k,l] - uRes2m1[i,j,k,l] - uRes2p1[i,j,k,l])*as_prefac_im +
                                        (2uRe[i,j,k,l] - uRes3m1[i,j,k,l] - uRes3p1[i,j,k,l])*as_prefac_im +
                                        VRe[i,j,k,l])
                - V_pre_fac_Re[i] * ( (2uIm[i,j,k,l] - uIms1m1[i,j,k,l] - uIms1p1[i,j,k,l])*as_prefac_re + 
                                        (2uIm[i,j,k,l] - uIms2m1[i,j,k,l] - uIms2p1[i,j,k,l])*as_prefac_re +
                                        (2uIm[i,j,k,l] - uIms3m1[i,j,k,l] - uIms3p1[i,j,k,l])*as_prefac_re +
                                        VIm[i,j,k,l])
                )
            
            AIm[i,j,k,l] = as^3 * pre_fac * (
                    (uRe[i,j,k,l] - uRetm1[i,j,k,l])*one_over_a_m1_Re[i] 
                - (uIm[i,j,k,l] - uImtm1[i,j,k,l])*one_over_a_m1_Im[i]  
                + (uRe[i,j,k,l] - uRetp1[i,j,k,l])*one_over_a_Re[i] 
                - (uIm[i,j,k,l] - uImtp1[i,j,k,l])*one_over_a_Im[i]
        
                - V_pre_fac_Re[i] * ( (2uRe[i,j,k,l] - uRes1m1[i,j,k,l] - uRes1p1[i,j,k,l])*as_prefac_re + 
                                        (2uRe[i,j,k,l] - uRes2m1[i,j,k,l] - uRes2p1[i,j,k,l])*as_prefac_re +  
                                        (2uRe[i,j,k,l] - uRes3m1[i,j,k,l] - uRes3p1[i,j,k,l])*as_prefac_re +    
                                        VRe[i,j,k,l])
                + V_pre_fac_Im[i] * ( (2uIm[i,j,k,l] - uIms1m1[i,j,k,l] - uIms1p1[i,j,k,l])*as_prefac_im +
                                        (2uIm[i,j,k,l] - uIms2m1[i,j,k,l] - uIms2p1[i,j,k,l])*as_prefac_im + 
                                        (2uIm[i,j,k,l] - uIms3m1[i,j,k,l] - uIms3p1[i,j,k,l])*as_prefac_im + 
                                    VIm[i,j,k,l])
                )
                #end
            #end
        end

        if isIdentity
            du[1:t_steps*n_steps^3,:] .= vec(ARe)
            du[t_steps*n_steps^3 + 1:end,:] .= vec(AIm)
        else
            _ARe = vec(ARe)
            _AIm = vec(AIm)
            KReARe .= KRe*_ARe
            KImAIm .= KIm*_AIm
            KReAIm .= KRe*_AIm
            KImARe .= KIm*_ARe
            @. du[1:t_steps*n_steps^3,:] = KReARe - KImAIm
            @. du[t_steps*n_steps^3 + 1:end,:] = KImARe + KReAIm
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









"""
    Calculate the drift loss used to update approximate the gradient
"""
function calcIMXLoss(sol_tr,KP::KernelProblem{ScalarField{2}}; H = KP.kernel.H)

    @unpack m, λ, contour, n_steps, as = KP.model
    @unpack a, t_steps = contour


    dt = 1e-5
    κ = 1e-2
    

    gm1=vcat([t_steps],1:t_steps-1)
    gp1=vcat(2:t_steps,[1])

    gsm1=vcat([n_steps],1:n_steps-1)
    gsp1=vcat(2:n_steps,[1])
    
    a_m1 = a[gm1]
    as_prefac_re = 1 / (as^2)
    as_prefac_im = 1/ (as^2) 

    one_over_a = a.^(-1)
    one_over_a_Re = real(one_over_a) #hcat([real(one_over_a) for i in 1:n_steps]...)
    one_over_a_Im = (imag(one_over_a) .- κ) #hcat([imag(one_over_a) for i in 1:n_steps]...)
    one_over_a_m1 = a_m1.^(-1)
    one_over_a_m1_Re = real(one_over_a_m1) #hcat([real(one_over_a_m1) for i in 1:n_steps]...)
    one_over_a_m1_Im = (imag(one_over_a_m1) .- κ) #hcat([imag(one_over_a_m1) for i in 1:n_steps]...)
    
    V_pre_fac = (a + a_m1)/2
    V_pre_fac_Re = real(V_pre_fac) #hcat([real(V_pre_fac) for i in 1:n_steps]...)
    V_pre_fac_Im = imag(V_pre_fac) #hcat([imag(V_pre_fac) for i in 1:n_steps]...)

    pre_fac = (1. / abs(a[1]))
    
    KRe,KIm = getK(H)
    KC = KRe .+ im*KIm
    
    im_pre_fac_KC = KC*im*pre_fac

    

    ARe = zeros(t_steps,n_steps,n_steps)
    AIm = zeros(t_steps,n_steps,n_steps)

    g(u) = begin

        _uRe = @view u[1:t_steps*n_steps^2]
        _uIm = @view u[t_steps*n_steps^2 + 1:end]
        
        uRe = reshape(_uRe,t_steps,n_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps,n_steps)

        _x = uRe + im * uIm
        for i in 1:3

            _ARe = Zygote.Buffer(ARe)
            _AIm = Zygote.Buffer(AIm)

            uRe = real(_x)
            uIm = imag(_x)

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

            VRe = @. m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
            VIm = @. m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)

            @inbounds for I in CartesianIndices((t_steps,n_steps,n_steps))
                i,j,k,l = Tuple(I)
                _ARe[i,j,k,l] = - as^2 * pre_fac * (
                        (uRe[i,j,k,l] - uRetm1[i,j,k,l])*one_over_a_m1_Im[i]
                    + (uIm[i,j,k,l] - uImtm1[i,j,k,l])*one_over_a_m1_Re[i]  
                    + (uRe[i,j,k,l] - uRetp1[i,j,k,l])*one_over_a_Im[i] 
                    + (uIm[i,j,k,l] - uImtp1[i,j,k,l])*one_over_a_Re[i]
    
                    - V_pre_fac_Im[i] * ( (2uRe[i,j,k,l] - uRes1m1[i,j,k,l] - uRes1p1[i,j,k,l])*as_prefac_im + 
                                            (2uRe[i,j,k,l] - uRes2m1[i,j,k,l] - uRes2p1[i,j,k,l])*as_prefac_im +
                                            VRe[i,j,k,l])
                    - V_pre_fac_Re[i] * ( (2uIm[i,j,k,l] - uIms1m1[i,j,k,l] - uIms1p1[i,j,k,l])*as_prefac_re + 
                                            (2uIm[i,j,k,l] - uIms2m1[i,j,k,l] - uIms2p1[i,j,k,l])*as_prefac_re +
                                            VIm[i,j,k,l])
                    )
                
                _AIm[i,j,k,l] = as^2 * pre_fac * (
                        (uRe[i,j,k,l] - uRetm1[i,j,k,l])*one_over_a_m1_Re[i] 
                    - (uIm[i,j,k,l] - uImtm1[i,j,k,l])*one_over_a_m1_Im[i]  
                    + (uRe[i,j,k,l] - uRetp1[i,j,k,l])*one_over_a_Re[i] 
                    - (uIm[i,j,k,l] - uImtp1[i,j,k,l])*one_over_a_Im[i]
            
                    - V_pre_fac_Re[i] * ( (2uRe[i,j,k,l] - uRes1m1[i,j,k,l] - uRes1p1[i,j,k,l])*as_prefac_re + 
                                            (2uRe[i,j,k,l] - uRes2m1[i,j,k,l] - uRes2p1[i,j,k,l])*as_prefac_re +    
                                            VRe[i,j,k,l])
                    + V_pre_fac_Im[i] * ( (2uIm[i,j,k,l] - uIms1m1[i,j,k,l] - uIms1p1[i,j,k,l])*as_prefac_im +
                                            (2uIm[i,j,k,l] - uIms2m1[i,j,k,l] - uIms2p1[i,j,k,l])*as_prefac_im + 
                                        VIm[i,j,k,l])
                    )
            end
            _A_tmp = copy(_ARe) + im*copy(_AIm)

            # _A_tmp =    @. ((_x - _x[gtm1,:]) / (a_m1 .- κ) + (_x - _x[gtp1,:]) / (a .- κ)  - 
            #                 (a + a_m1)/2 * ( (2*_x - _x[:,gsm1] - _x[:,gsp1]) * as_prefac 
            #                     +  (m * _x + (λ/6) * _x^3)))

            _A = im_pre_fac_KC * vec(_A_tmp)

            _x += reshape(_A * dt,t_steps,n_steps,n_steps)
        end

        return _x
        #return sum(imag(_x).^2) + sum(real(_x).^2)

    end

    XX = [g(u) for u in eachrow(sol_tr')]

    xRe = sum( abs2.(StatsBase.mean([real(_x) for _x in XX]) ) )
    xIm = sum( abs2.(StatsBase.mean([imag(_x) for _x in XX]) ) )

    x2Re = "phi2Re" in keys(KP.y) ? sum( abs2.(StatsBase.mean([real(_x).^2 .- imag(_x).^2 for _x in XX]) .- KP.y["phi2Re"]) ) : 0.
    x2Im = sum( abs2.(StatsBase.mean([2 .* real(_x) .* imag(_x) for _x in XX]) .- KP.y["phi2Im"]) )

    imx = sum( abs.(StatsBase.mean([imag(_x).^2 + real(_x).^2 for _x in XX])) )

    return xRe + xIm + 10*x2Re + x2Im + imx

    # return sum(
    #     mean(g(u) for u in eachrow(sol_tr'))
    #     )

end


