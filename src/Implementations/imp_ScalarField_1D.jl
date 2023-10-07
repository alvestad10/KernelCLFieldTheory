

"""
    Get the drift and noise term to be used in the simulation
"""
function get_ab(model::ScalarField{1},kernel::T) where {T <: ConstantKernel}
    
    @unpack m, λ, contour, n_steps, as = model
    @unpack a, t_steps = contour
    @unpack H = kernel

    _KRe, _KIm = getK(kernel)
    KRe = Matrix(_KRe)
    KIm = Matrix(_KIm)

    use_GPU = false
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

    pre_fac = (1. / (as*abs(a[1])))
    #im_pre_fac = im*pre_fac

    isIdentity = all(diag(H) .- 1. .== 0.0)
    #tmp = zeros(8t_steps,n_steps)
    #dtmp = dualcache(tmp)
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
        
        if !isIdentity
            KReARe = vec(@view tmp[4t_steps .+ (1:t_steps),:])
            KImAIm = vec(@view tmp[5t_steps .+ (1:t_steps),:])
            KReAIm = vec(@view tmp[6t_steps .+ (1:t_steps),:])
            KImARe = vec(@view tmp[7t_steps .+ (1:t_steps),:])
        end

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
        

        @inbounds for I in CartesianIndices((t_steps,n_steps))
            i,j = Tuple(I)
            ARe[i,j] = - as * pre_fac * (
                    (uRe[i,j] - uRetm1[i,j])*one_over_a_m1_Im[i]
                + (uIm[i,j] - uImtm1[i,j])*one_over_a_m1_Re[i]  
                + (uRe[i,j] - uRetp1[i,j])*one_over_a_Im[i] 
                + (uIm[i,j] - uImtp1[i,j])*one_over_a_Re[i]

                - V_pre_fac_Im[i] * ( (2uRe[i,j] - uResm1[i,j] - uResp1[i,j])*as_prefac_im + 
                                        VRe[i,j])
                - V_pre_fac_Re[i] * ( (2uIm[i,j] - uImsm1[i,j] - uImsp1[i,j])*as_prefac_re + 
                                        VIm[i,j])
                )
            
            AIm[i,j] = as * pre_fac * (
                    (uRe[i,j] - uRetm1[i,j])*one_over_a_m1_Re[i] 
                - (uIm[i,j] - uImtm1[i,j])*one_over_a_m1_Im[i]  
                + (uRe[i,j] - uRetp1[i,j])*one_over_a_Re[i] 
                - (uIm[i,j] - uImtp1[i,j])*one_over_a_Im[i]
        
                - V_pre_fac_Re[i] * ( (2uRe[i,j] - uResm1[i,j] - uResp1[i,j])*as_prefac_re + 
                                        VRe[i,j])
                + V_pre_fac_Im[i] * ( (2uIm[i,j] - uImsm1[i,j] - uImsp1[i,j])*as_prefac_im +
                                    VIm[i,j])
                )
        end

        if isIdentity
            du[1:t_steps*n_steps,:] .= vec(ARe)
            du[t_steps*n_steps + 1:end,:] .= vec(AIm)
        elseif use_GPU
            _ARe = CuArray(vec(ARe))
            _AIm = CuArray(vec(AIm))
            
            copyto!(du[1:t_steps*n_steps,:],KRe*_ARe - KIm*_AIm)
            copyto!(du[t_steps*n_steps + 1:end,:],KIm*_ARe + KRe*_AIm)
        else
            _ARe = vec(ARe)
            _AIm = vec(AIm)
            KReARe .= KRe*_ARe
            KImAIm .= KIm*_AIm
            KReAIm .= KRe*_AIm
            KImARe .= KIm*_ARe

            # mul!(KReARe, KRe,_ARe)
            # mul!(KImAIm, KIm,_AIm)
            # mul!(KReAIm, KRe,_AIm)
            # mul!(KImARe, KIm,_ARe)
            @. du[1:t_steps*n_steps,:] = KReARe - KImAIm
            @. du[t_steps*n_steps + 1:end,:] = KImARe + KReAIm
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
#     Get the drift and noise term to be used in the simulation
# """
# function get_ab(model::ScalarField{1},kernel::T) where {T <: ConstantKernel}
    
#     @unpack m, λ, contour, n_steps, as = model
#     @unpack a, t_steps = contour
#     @unpack H = kernel

#     _KRe, _KIm = getK(kernel)
#     KRe = Matrix(Float32.(_KRe))
#     KIm = Matrix(Float32.(_KIm))

#     KRe = cu(KRe)
#     KIm = cu(KIm)

#     gm1=vcat([t_steps],1:t_steps-1)
#     gp1=vcat(2:t_steps,[1])

#     gsm1=vcat([n_steps],1:n_steps-1)
#     gsp1=vcat(2:n_steps,[1])
    
#     a = a
#     a_m1 = a[gm1]
#     as_prefac_re = 1 / (as^2)
#     as_prefac_im = 1/ (as^2) 

#     one_over_a = a.^(-1)
#     one_over_a_Re = cu(Float32.(real(one_over_a))) #hcat([real(one_over_a) for i in 1:n_steps]...)
#     one_over_a_Im = cu(Float32.(imag(one_over_a))) #hcat([imag(one_over_a) for i in 1:n_steps]...)
#     one_over_a_m1 = a_m1.^(-1)
#     one_over_a_m1_Re = cu(Float32.(real(one_over_a_m1))) #hcat([real(one_over_a_m1) for i in 1:n_steps]...)
#     one_over_a_m1_Im = cu(Float32.(imag(one_over_a_m1))) #hcat([imag(one_over_a_m1) for i in 1:n_steps]...)
    
#     V_pre_fac = (a + a_m1)/2
#     V_pre_fac_Re = cu(Float32.(real(V_pre_fac))) #hcat([real(V_pre_fac) for i in 1:n_steps]...)
#     V_pre_fac_Im = cu(Float32.(imag(V_pre_fac))) #hcat([imag(V_pre_fac) for i in 1:n_steps]...)

#     pre_fac = Float32(1. / (abs(a[1])*as))
#     #im_pre_fac = im*pre_fac

#     isIdentity = all(diag(H) .- 1. .== 0.0)

#     tmp1 = cu(zeros(Float32,t_steps,n_steps))
#     tmp2 = cu(zeros(Float32,t_steps,n_steps))
#     tmp3 = cu(zeros(Float32,t_steps,n_steps))
#     tmp4 = cu(zeros(Float32,t_steps,n_steps))
#     tmp5 = cu(zeros(Float32,t_steps*n_steps))
#     tmp6 = cu(zeros(Float32,t_steps*n_steps))
#     tmp7 = cu(zeros(Float32,t_steps*n_steps))
#     tmp8 = cu(zeros(Float32,t_steps*n_steps))
#     #tmp = cu(tmp)
#     dtmp1 = dualcache(tmp1)
#     dtmp2 = dualcache(tmp2)
#     dtmp3 = dualcache(tmp3)
#     dtmp4 = dualcache(tmp4)
#     dtmp5 = dualcache(tmp5)
#     dtmp6 = dualcache(tmp6)
#     dtmp7 = dualcache(tmp7)
#     dtmp8 = dualcache(tmp8)

#     @inbounds function a_func!(du,u,p,t)
#         _uRe = @view u[1:t_steps*n_steps]
#         _uIm = @view u[t_steps*n_steps + 1:end]

#         uRe = reshape(_uRe,t_steps,n_steps)
#         uIm = reshape(_uIm,t_steps,n_steps)


#         uRetm1 = @view uRe[gm1,:]
#         uRetp1 = @view uRe[gp1,:]
#         uResm1 = @view uRe[:,gsm1]
#         uResp1 = @view uRe[:,gsp1]
        
#         uImtm1 = @view uIm[gm1,:]
#         uImtp1 = @view uIm[gp1,:]
#         uImsm1 = @view uIm[:,gsm1]
#         uImsp1 = @view uIm[:,gsp1]


#         #tmp = get_tmp(dtmp, u)
#         ARe = get_tmp(dtmp1, u) #@view tmp[1:t_steps,:]
#         AIm = get_tmp(dtmp2, u) #@view tmp[t_steps .+ (1:t_steps),:]
#         VRe = get_tmp(dtmp3, u) #@view tmp[2t_steps .+ (1:t_steps),:]
#         VIm = get_tmp(dtmp4, u) #@view tmp[3t_steps .+ (1:t_steps),:]
        
#         if !isIdentity
#             KReARe = get_tmp(dtmp5, u) #vec(@view tmp[4t_steps .+ (1:t_steps),:])
#             KImAIm = get_tmp(dtmp6, u) #vec(@view tmp[5t_steps .+ (1:t_steps),:])
#             KReAIm = get_tmp(dtmp7, u) #vec(@view tmp[6t_steps .+ (1:t_steps),:])
#             KImARe = get_tmp(dtmp8, u) #vec(@view tmp[7t_steps .+ (1:t_steps),:])
#         end
        
#         @. VRe = m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
#         @. VIm = m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)
        
#         @. ARe = - as * pre_fac * (
#                           (uRe - uRetm1) * one_over_a_m1_Im
#                         + (uIm - uImtm1) * one_over_a_m1_Re  
#                         + (uRe - uRetp1) * one_over_a_Im 
#                         + (uIm - uImtp1) * one_over_a_Re

#                         -  ( (2uRe - uResm1 - uResp1) * as_prefac_im + 
#                             VRe) * V_pre_fac_Im
#                         - ( (2uIm - uImsm1 - uImsp1) * as_prefac_re + 
#                             VIm) * V_pre_fac_Re
#         )

#         @. AIm = as * pre_fac * (
#                          (uRe - uRetm1) * one_over_a_m1_Re 
#                         - (uIm - uImtm1) * one_over_a_m1_Im  
#                         + (uRe - uRetp1) * one_over_a_Re 
#                         - (uIm - uImtp1) * one_over_a_Im

#                         - ( (2uRe - uResm1 - uResp1)*as_prefac_re + 
#                             VRe) * V_pre_fac_Re
#                         + ( (2uIm - uImsm1 - uImsp1)*as_prefac_im +
#                             VIm) * V_pre_fac_Im
#         )

#         if isIdentity
#             du[1:t_steps*n_steps,:] .= vec(ARe)
#             du[t_steps*n_steps + 1:end,:] .= vec(AIm)
#         else
#             _ARe = vec(ARe)
#             _AIm = vec(AIm)
#             #println(typeof(_ARe))
#             #println(typeof(KReARe))
#             # KReARe .= KRe*_ARe
#             # KImAIm .= KIm*_AIm
#             # KReAIm .= KRe*_AIm
#             # KImARe .= KIm*_ARe

#             mul!(KReARe, KRe,_ARe)
#             mul!(KImAIm, KIm,_AIm)
#             mul!(KReAIm, KRe,_AIm)
#             mul!(KImARe, KIm,_ARe)
#             @. du[1:t_steps*n_steps,:] = KReARe - KImAIm
#             @. du[t_steps*n_steps + 1:end,:] = KImARe + KReAIm
#         end
#         nothing
#     end
    
    
#     sqrt2pre_fac = cu(sqrt(2 * pre_fac))
#     sqrt2pre_fac_sqrtK = cu(sqrt(2 * pre_fac) * H)
    
#     function b_func!(du,u,p,t)
#         if isIdentity
#             @. du[1:div(end,2)] = sqrt2pre_fac
#         else
#             @. du = sqrt2pre_fac_sqrtK
#         end
#         nothing
#     end 

#     return a_func!, b_func!
# end












# """
#     Get the drift and noise term to be used in the simulation
# """
# function get_ab(model::ScalarField{1},kernel::T) where {T <: ConstantKernel}
    
#     @unpack m, λ, contour, n_steps, as = model
#     @unpack a, t_steps = contour
#     @unpack H = kernel

#     _KRe, _KIm = getK(kernel)
#     KRe = Matrix(_KRe)
#     KIm = Matrix(_KIm)

#     KRe = cu(KRe)
#     KIm = cu(KIm)

#     gm1=vcat([t_steps],1:t_steps-1)
#     gp1=vcat(2:t_steps,[1])

#     gsm1=vcat([n_steps],1:n_steps-1)
#     gsp1=vcat(2:n_steps,[1])
    
#     a = a
#     a_m1 = a[gm1]
#     as_prefac_re = 1 / (as^2)
#     as_prefac_im = 1/ (as^2) 

#     one_over_a = a.^(-1)
#     one_over_a_Re = cu(real(one_over_a)) #hcat([real(one_over_a) for i in 1:n_steps]...)
#     one_over_a_Im = cu(imag(one_over_a)) #hcat([imag(one_over_a) for i in 1:n_steps]...)
#     one_over_a_m1 = a_m1.^(-1)
#     one_over_a_m1_Re = cu(real(one_over_a_m1)) #hcat([real(one_over_a_m1) for i in 1:n_steps]...)
#     one_over_a_m1_Im = cu(imag(one_over_a_m1)) #hcat([imag(one_over_a_m1) for i in 1:n_steps]...)
    
#     V_pre_fac = (a + a_m1)/2
#     V_pre_fac_Re = cu(real(V_pre_fac)) #hcat([real(V_pre_fac) for i in 1:n_steps]...)
#     V_pre_fac_Im = cu(imag(V_pre_fac)) #hcat([imag(V_pre_fac) for i in 1:n_steps]...)

#     pre_fac = (1. / (abs(a[1])*as))
#     #im_pre_fac = im*pre_fac

#     isIdentity = all(diag(H) .- 1. .== 0.0)

#     function a_func!(du,u,p,t)

#         @inbounds(begin
#         _uRe = @view u[1:t_steps*n_steps]
#         _uIm = @view u[t_steps*n_steps + 1:end]

#         uRe = reshape(_uRe,t_steps,n_steps)
#         uIm = reshape(_uIm,t_steps,n_steps)


#         #uRe = @view _u[1:t_steps,:]
#         uRetm1 = @view uRe[gm1,:]
#         uRetp1 = @view uRe[gp1,:]
#         uResm1 = @view uRe[1:t_steps,gsm1]
#         uResp1 = @view uRe[1:t_steps,gsp1]
        
#         #uIm = @view _u[t_steps+1:end,:]
#         uImtm1 = @view uIm[gm1,:]
#         uImtp1 = @view uIm[gp1,:]
#         uImsm1 = @view uIm[1:t_steps,gsm1]
#         uImsp1 = @view uIm[1:t_steps,gsp1]
        
#         @tullio VRe[i,j] := m * uRe[i,j] + (λ/6) * (uRe[i,j]^3 - 3*uRe[i,j]*uIm[i,j]^2)
#         @tullio VIm[i,j] := m * uIm[i,j] - (λ/6) * (uIm[i,j]^3 - 3*uIm[i,j]*uRe[i,j]^2)
        
#         @tullio ARe[i,j] := - as * pre_fac * (
#                             (uRe[i,j] - uRetm1[i,j])*one_over_a_m1_Im[i]
#                         + (uIm[i,j] - uImtm1[i,j])*one_over_a_m1_Re[i]  
#                         + (uRe[i,j] - uRetp1[i,j])*one_over_a_Im[i] 
#                         + (uIm[i,j] - uImtp1[i,j])*one_over_a_Re[i]

#                         - V_pre_fac_Im[i] * ( (2uRe[i,j] - uResm1[i,j] - uResp1[i,j])*as_prefac_im + 
#                                                 VRe[i,j])
#                         - V_pre_fac_Re[i] * ( (2uIm[i,j] - uImsm1[i,j] - uImsp1[i,j])*as_prefac_re + 
#                                                 VIm[i,j])
#         )

#         @tullio AIm[i,j] := as * pre_fac * (
#                             (uRe[i,j] - uRetm1[i,j])*one_over_a_m1_Re[i] 
#                         - (uIm[i,j] - uImtm1[i,j])*one_over_a_m1_Im[i]  
#                         + (uRe[i,j] - uRetp1[i,j])*one_over_a_Re[i] 
#                         - (uIm[i,j] - uImtp1[i,j])*one_over_a_Im[i]

#                         - V_pre_fac_Re[i] * ( (2uRe[i,j] - uResm1[i,j] - uResp1[i,j])*as_prefac_re + 
#                                                 VRe[i,j])
#                         + V_pre_fac_Im[i] * ( (2uIm[i,j] - uImsm1[i,j] - uImsp1[i,j])*as_prefac_im +
#                                             VIm[i,j])
#         )

#         _ARe = vec(ARe)
#         _AIm = vec(AIm)

#         duRe = @view du[1:t_steps*n_steps]
#         duIm = @view du[t_steps*n_steps + 1:end]

#         @tullio duRe[i] = KRe[i,j]*_ARe[j] - KIm[i,j]*_AIm[j] 
#         @tullio duIm[i] = KIm[i,j]*_ARe[j] + KRe[i,j]*_AIm[j]
#         end)
#         nothing
#     end
    
    
#     sqrt2pre_fac = sqrt(2 * pre_fac)
#     sqrt2pre_fac_sqrtK = cu(sqrt(2 * pre_fac) * H)
    
#     function b_func!(du,u,p,t)
#         if isIdentity
#             @. du[1:div(end,2)] = sqrt2pre_fac
#         else
#             @. du = sqrt2pre_fac_sqrtK
#         end
#         nothing
#     end 

#     return a_func!, b_func!
# end




# """
#     Caluclate the true loss
# """
# function calcTrueLoss(sol,KP::KernelProblem{}; obs_calc = nothing)
    
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
function calcIMXLoss(sol_tr,KP::KernelProblem{ScalarField{1}}; H = KP.kernel.H)

    @unpack m, λ, contour, n_steps, as = KP.model
    @unpack a, t_steps = contour
    

    gtm1=vcat([t_steps],1:t_steps-1)
    gtp1=vcat(2:t_steps,[1])

    gsm1=vcat([n_steps],1:n_steps-1)
    gsp1=vcat(2:n_steps,[1])

    dt = 1e-5
    κ = im*1e-4
    
    a_m1 = a[gtm1]
    as_prefac = 1 / (as^2) 

    one_over_a = 1 ./ (a .- κ)
    one_over_am1 = 1 ./ (a_m1 .- κ)
    
    pre_fac = (as / abs(a[1]))
    
    KRe,KIm = getK(H)
    KC = KRe .+ im*KIm

    a_ap1_fac = (a .+ a_m1)/2
    
    im_pre_fac_KC = KC*im*pre_fac

    

    g(u) = begin

        _uRe = @view u[1:t_steps*n_steps]
        _uIm = @view u[t_steps*n_steps + 1:end]
        
        uRe = reshape(_uRe,t_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps)
        
        _x = uRe + im * uIm
        _xtm1 = @view _x[gtm1,:]
        _xtp1 = @view _x[gtp1,:]
        _xsm1 = @view _x[:,gsm1]
        _xsp1 = @view _x[:,gsp1]

        for i in 1:5

            @tullio _A_tmp[i,j] := ((_x[i,j] - _xtm1[i,j]) * one_over_am1[i]  + (_x[i,j] - _xtp1[i,j]) * one_over_a[i]  - 
                    a_ap1_fac[i] * ( (2*_x[i,j] - _xsm1[i,j] - _xsp1[i,j]) * as_prefac 
                                +  (m * _x[i,j] + (λ/6) * _x[i,j]^3)))

            _A = im_pre_fac_KC * vec(_A_tmp)

            _x += reshape(_A * dt,t_steps,n_steps)
        end

        
        return _x
        #return sum(imag(_x).^2) # + sum(real(_x).^2)

    end

    XX = [g(u) for u in eachrow(sol_tr')]
    # XP0 = [mean(X,dims=2) for X in XX]

    # xRe = sum( abs2.(StatsBase.mean([real(X) for X in XP0]) ) )
    # xIm = sum( abs2.(StatsBase.mean([imag(X) for X in XP0]) ) )

    # x2Re = sum( abs2.(StatsBase.mean([real(X).^2 .- imag(X).^2 for X in XP0]) .- KP.y["phi2Re"]) )
    # x2Im = sum( abs2.(StatsBase.mean([2 .* real(X) .* imag(X) for X in XP0]) .- KP.y["phi2Im"]) )

    imx = sum([maximum(imag(_x).^2) for _x in XX])
    # rex = sum([maximum(real(_x).^2) for _x in XX])

    return imx
    # return xRe + xIm + 5*x2Re + x2Im + imx + rex
    # return sum(
    #     mean(g(u) for u in eachrow(sol_tr'))
    #     )

end