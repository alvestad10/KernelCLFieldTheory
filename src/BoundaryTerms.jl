
export getBoundaryTerms, calcBoundaryTerms, calcBoundaryTermsCorrections

#######################################
abstract type AbstractBoundaryTerms end

struct BoundaryTerms{MType,YT,LType,L2Type} <: AbstractBoundaryTerms
    model::MType
    Ys::Vector{YT}
    Xs::Union{Vector{YT},Nothing}
    L_CO::LType
    L2_CO::L2Type
    function BoundaryTerms(model::MType,Ys::Vector{YT},L_CO::LType) where {MType <: Model,YT,LType}
        new{MType,YT,LType,Nothing}(model,Ys,nothing,L_CO,nothing)
    end
    function BoundaryTerms(model::MType,Ys::Vector{YT},Xs::Union{Vector{YT},Nothing},L_CO::LType) where {MType <: Model,YT,LType}
        new{MType,YT,LType,Nothing}(model,Ys,Xs,L_CO,nothing)
    end
    function BoundaryTerms(model::MType,Ys::Vector{YT},L_CO::LType,L2_CO::L2Type) where {MType <: Model,YT,LType,L2Type}
        new{MType,YT,LType,L2Type}(model,Ys,nothing,L_CO,L2_CO)
    end
    function BoundaryTerms(model::MType,Ys::Vector{YT},Xs::Union{Vector{YT},Nothing},L_CO::LType,L2_CO::L2Type) where {MType <: Model,YT,LType,L2Type}
        new{MType,YT,LType,L2Type}(model,Ys,Xs,L_CO,L2_CO)
    end
end




function Lc_x_x2_x0xt(u,H,m,λ,a,a_m1,gp1,gm1,κ)

    #@unpack K = kernel.pK
    KRe,KIm = getK(H)

    Lcx = zeros(size(u))
    Lcx2 = zeros(size(u))
    Lcx0xt = zeros(size(u))

    #_A = similar(u)
    #_B = similar(u)

    xR =  @view u[1:div(end,2)]
    xI =  @view u[div(end,2)+1:end]
    
    #x = xR .+ im*xI

    KreT = transpose(KRe)
    KimT = transpose(KIm)

    pre_fac = (1 / abs(a[1]))
    
    #_A[1:div(end,2)] 
    ARe = - pre_fac .* 0.5 .* (
        2. .* ( real.(a_m1) .* (xI .- xI[gp1])  .- imag.(a_m1 .- κ).*(xR .- xR[gp1]) ) ./ abs.(a_m1).^2
    .- 2. .* ( real.(a)  .* (xI[gm1] .- xI) .- imag.(a .- κ)  .* (xR[gm1] .- xR) )./ abs.(a).^2
    .- real.(a_m1 .+ a) .* (m .* xI .+ (1/6)*λ .* (-(xI.^3) .+ 3xI.*(xR.^2))) 
    .- imag.(a_m1 .+ a) .* (m .* xR .+ (1/6)*λ .* (xR.^3 .- 3xR.*(xI.^2))) 
    )     
    
    #_A[div(end,2)+1:end] 
    AIm = pre_fac .* 0.5 .* (
        2. .* (real.(a_m1).*(xR .- xR[gp1]) .+ imag.(a_m1 .- κ).*(xI .- xI[gp1])) ./ abs.(a_m1).^2
    .- 2. .* (real.(a).*(xR[gm1] .- xR)   .+ imag.(a .- κ).*(xI[gm1] .- xI)) ./ abs.(a).^2
    .- real.(a_m1 .+ a) .* (m.*xR .+ (1/6)*λ .* ((xR.^3) .- 3xR.*(xI.^2))) 
    .+ imag.(a_m1 .+ a) .* (m.*xI .+ (1/6)*λ .* (-xI.^3 .+ 3xI.*(xR.^2))) 
    )

    SjKTRe = KRe*ARe .- KIm*AIm
    SjKTIm = KRe*AIm .+ KIm*ARe
    
    Lcx[1:div(end,2)]     .= SjKTRe  #vec((transpose(ARe)*kReT .- transpose(AIm)*kImT))
    Lcx[div(end,2)+1:end] .= SjKTIm #vec((transpose(AIm)*kReT .+ transpose(ARe)*kImT))

    Lcx2[1:div(end,2)]     .= pre_fac*2*diag(KreT) .+ 2*(SjKTRe .* xR .- SjKTIm .* xI)
    Lcx2[div(end,2)+1:end] .= pre_fac*2*diag(KimT) .+ 2*(SjKTRe .* xI .+ SjKTIm .* xR)
    
    SjKj0Re = ARe.*view(KreT,:,1) .- AIm.*view(KimT,:,1)
    SjKj0Im = AIm.*view(KreT,:,1) .+ ARe.*view(KimT,:,1)
    Lcx0xt[1:div(end,2)]     .= pre_fac*(vec(KreT[1,:]) .+ vec(KreT[:,1])) .+ (SjKTRe .* xR[1] .- SjKTIm .* xI[1]) .+ (SjKj0Re .* xR .- SjKj0Im .* xI)
    Lcx0xt[div(end,2)+1:end] .= pre_fac*(vec(KimT[1,:]) .+ vec(KimT[:,1])) .+ (SjKTRe .* xI[1] .+ SjKTIm .* xR[1]) .+ (SjKj0Re .* xI .+ SjKj0Im .* xR)
    return [Lcx...,Lcx2...,Lcx0xt...]
end

function getBoundaryTermsObservables(KP::KernelProblem{ScalarField{0}};kwargs...)

    @unpack model, kernel = KP
    @unpack m, λ, contour = model
    @unpack a, t_steps = contour
    H = kernel.H

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]

    κ = 0. * im

    LcO(x) = Lc_x_x2_x0xt(x,H,m,λ,a,a_m1,gp1,gm1,κ)
    return LcO, nothing#L2cO
end


function getBoundaryTermsObservables(KP::KernelProblem{ScalarField{1}};kwargs...)

    @unpack model, kernel = KP
    @unpack m, λ, contour, n_steps, as = model
    @unpack a, t_steps = contour
    @unpack H = kernel

    _KRe, _KIm = getK(kernel)
    KRe = Matrix(_KRe)
    KIm = Matrix(_KIm)

    KRe_tensor = reshape(transpose(KRe),t_steps,n_steps,t_steps,n_steps)
    KIm_tensor = reshape(transpose(KIm),t_steps,n_steps,t_steps,n_steps)

    diagKRe = diag(reshape(mean(KRe_tensor,dims=(2,4)),t_steps,t_steps)) ./ n_steps
    diagKIm = diag(reshape(mean(KIm_tensor,dims=(2,4)),t_steps,t_steps)) ./ n_steps

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

    pre_fac = (1. / abs(a[1]))
    #im_pre_fac = im*pre_fac

    isIdentity = all(diag(H) .- 1. .== 0.0)

    LcO(u) = begin
    
        @unpack m, λ, contour, n_steps, as = model
        @unpack t_steps = contour
    
        Lcx = zeros(2*t_steps)
        Lcx2 = zeros(2*t_steps)
        Lcx0xt = zeros(2*t_steps)
    
        _uRe = @view u[1:t_steps*n_steps]
        _uIm = @view u[t_steps*n_steps + 1:end]

        uRe = reshape(_uRe,t_steps,n_steps)
        uIm = reshape(_uIm,t_steps,n_steps)
        ARe = zeros(t_steps,n_steps)
        AIm = zeros(t_steps,n_steps)

        uRetm1 = @view uRe[gm1,:]
        uRetp1 = @view uRe[gp1,:]
        uResm1 = @view uRe[1:t_steps,gsm1]
        uResp1 = @view uRe[1:t_steps,gsp1]
        
        uImtm1 = @view uIm[gm1,:]
        uImtp1 = @view uIm[gp1,:]
        uImsm1 = @view uIm[1:t_steps,gsm1]
        uImsp1 = @view uIm[1:t_steps,gsp1]
        
        VRe = @. m * uRe + (λ/6) * (uRe^3 - 3*uRe*uIm^2)
        VIm = @. m * uIm - (λ/6) * (uIm^3 - 3*uIm*uRe^2)
 

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
            SjKTRe = vec(ARe)
            SjKTIm = vec(AIm)
        elseif use_GPU
            _ARe = CuArray(vec(ARe))
            _AIm = CuArray(vec(AIm))
            
            SjKTRe = Array(KRe*_ARe - KIm*_AIm)
            SjKTIm = Array(KIm*_ARe + KRe*_AIm)
        else
            _ARe = vec(ARe)
            _AIm = vec(AIm)
            KReARe = KRe*_ARe
            KImAIm = KIm*_AIm
            KReAIm = KRe*_AIm
            KImARe = KIm*_ARe

            SjKTRe = KReARe - KImAIm
            SjKTIm = KImARe + KReAIm
        end

        SjKTRe = mean(reshape(SjKTRe,t_steps,n_steps), dims=2)
        SjKTIm = mean(reshape(SjKTIm,t_steps,n_steps), dims=2)
        
        Lcx[1:div(end,2)]     .= SjKTRe #mean(SjKTRe,dims=2)  #vec((transpose(ARe)*kReT .- transpose(AIm)*kImT))
        Lcx[div(end,2)+1:end] .= SjKTIm #mean(SjKTIm,dims=2) #vec((transpose(AIm)*kReT .+ transpose(ARe)*kImT))
        
        m_uRe = mean(uRe,dims=2)
        m_uIm = mean(uIm,dims=2)

        Lcx2[1:div(end,2)]     .= pre_fac*2*diagKRe .+ 2*(SjKTRe .* m_uRe .- SjKTIm .* m_uIm) ./ n_steps
        Lcx2[div(end,2)+1:end] .= pre_fac*2*diagKIm .+ 2*(SjKTRe .* m_uIm .+ SjKTIm .* m_uRe) ./ n_steps
        
        return [Lcx...,Lcx2...,Lcx0xt...]
    end
    
    return LcO, nothing#L2cO
end



function getBoundaryTerms(KP::KernelProblem;Ys=nothing,Xs=nothing,kwargs...)

    @unpack model, kernel = KP

    if isnothing(Ys)
        Ys = collect(0:0.1:8)
    end

    L_CO,L2_CO = getBoundaryTermsObservables(KP;kwargs...)
    return BoundaryTerms(model,Ys,Xs,L_CO,L2_CO)
end



function calcBoundaryTerms(sol,BT::BoundaryTerms{MType,YT,L,L2}) where {MType <: Model,YT,L,L2}
    
    @unpack Ys = BT
    NTr = length(sol)

    NObs = 6*BT.model.contour.t_steps

    LOs_Omega = zeros(T,NObs,length(Ys),NTr)
    LOs_Y = zeros(T,NObs,length(Ys),NTr)
    #LOs_err = zeros(T,NObs,length(Ys),NTr)

    Threads.@threads for tr in 1:NTr
    #for tr in 1:NTr

        N = size(sol[tr])[2]
        trVs = zeros(T,NObs,N)
        #dt = sol[tr].t[end] .- sol[tr].t[end-1]
        #nn = 0.1 / dt
        
        @inbounds @simd for i in 1:N
            tri = @view sol[tr][:,i]
            trVs[:,i] = BT.L_CO(tri)
        end
        

        ### TODO: This needs to be abstracted away to the specific model
        XX = @view sol[tr][1:div(end,2),:]
        YY = @view sol[tr][div(end,2)+1:end,:]
        
        #Omega = [sum(abs.(XX.^2 .- YY.^2),dims=1) ; sum(abs.(2 .* XX .* YY),dims=1)]
        Omega = [maximum(abs.(XX),dims=1) ; maximum(abs.(YY),dims=1)]
        #Omega = [mean(abs.(XX),dims=1) ; mean(abs.(YY),dims=1)]
    

        trLOs_Omega = zeros(T,NObs,length(Ys))
        trLOs_Y = zeros(T,NObs,length(Ys))
        #err_trLOs = zeros(T,NObs,length(Ys))
        for i in eachindex(Ys)
            Hinx_Omega = @. (Omega[2,:] .<= Ys[i]) .& (Omega[1,:] .<= Ys[i])
            Hinx_Y = @. (Omega[2,:] .<= Ys[i]) 
            
            H_Omega = @view trVs[:,Hinx_Omega]
            H_Y = @view trVs[:,Hinx_Y]
            
            trLOs_Omega[:,i] = sum(H_Omega,dims=2)/N
            trLOs_Y[:,i] = sum(H_Y,dims=2)/N

            #err_trLOs[:,i] = sqrt.(sum(H.^2,dims=2)/N .- trLOs[:,i].^2) ./ sqrt(N/nn)
        end
        LOs_Omega[:,:,tr] .= trLOs_Omega
        LOs_Y[:,:,tr] .= trLOs_Y
        #LOs_err[:,:,tr] .= err_trLOs
    end
    
    #DD = NTr * sum( 1 ./ LOs_err.^2, dims=3)
    #return mean(LOs,dims=3)  .± sqrt.( 1 ./ DD )
    
    BT_Omega = mean(LOs,dims=3) .± std(LOs,dims=3) ./ sqrt(NTr)
    BT_Y = mean(LOs,dims=3) .± std(LOs,dims=3) ./ sqrt(NTr)
    return BT_Omega, BT_Y
end


