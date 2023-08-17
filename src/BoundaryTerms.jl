
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

    #A = im*pre_fac*( 
    #            (x .- x[gp1]) ./ a_m1 + (x .- x[gm1]) ./ a
    #            .- (a .+ a_m1)/2 .* (m .* x .+ (λ/6) .* x.^3)
    #    )
    #ARe = real(A)
    #AIm = imag(A)

    SjKTRe = KRe*ARe .- KIm*AIm
    SjKTIm = KRe*AIm .+ KIm*ARe
    
    #mul!(_B,K,[ARe ; AIm])
    #_B = _A#transpose(_B)

    #kReT = (p[1:div(end,2),:]*transpose(p[1:div(end,2),:]) 
    #            - p[div(end,2)+1:end,:]*transpose(p[div(end,2)+1:end,:]))
    #kImT = (p[1:div(end,2),:]*transpose(p[div(end,2)+1:end,:]) 
    #            + p[div(end,2)+1:end,:]*transpose(p[1:div(end,2),:]))
    
    #ARe = @view _A[1:div(end,2)]
    #AIm = @view _A[div(end,2)+1:end]
    #SjKTRe = vec(transpose(ARe)*KreT .- transpose(AIm)*KimT)
    #SjKTRe = @view _B[1:div(end,2)]#ARe #vec(transpose(ARe)*kReT .- transpose(AIm)*kImT)
    #SjKTIm = vec(transpose(AIm)*KreT .+ transpose(ARe)*KimT)
    #SjKTIm = @view _B[div(end,2) + 1:end]#AIm #vec(transpose(AIm)*kReT .+ transpose(ARe)*kImT)

    Lcx[1:div(end,2)]     .= SjKTRe  #vec((transpose(ARe)*kReT .- transpose(AIm)*kImT))
    Lcx[div(end,2)+1:end] .= SjKTIm #vec((transpose(AIm)*kReT .+ transpose(ARe)*kImT))
    

    
    
    Lcx2[1:div(end,2)]     .= pre_fac*2*diag(KreT) .+ 2*(SjKTRe .* xR .- SjKTIm .* xI)
    Lcx2[div(end,2)+1:end] .= pre_fac*2*diag(KimT) .+ 2*(SjKTRe .* xI .+ SjKTIm .* xR)
    
    Lcx2[1:1]     .= sum(pre_fac*2*diag(KreT) .+ 2*(SjKTRe .* xR .- SjKTIm .* xI))
    Lcx2[div(end,2)+1:div(end,2)+1] .= sum(pre_fac*2*diag(KimT) .+ 2*(SjKTRe .* xI .+ SjKTIm .* xR))
    

    
    SjKj0Re = ARe.*view(KreT,:,1) .- AIm.*view(KimT,:,1)
    SjKj0Im = AIm.*view(KreT,:,1) .+ ARe.*view(KimT,:,1)
    Lcx0xt[1:div(end,2)]     .= pre_fac*(vec(KreT[1,:]) .+ vec(KreT[:,1])) .+ (SjKTRe .* xR[1] .- SjKTIm .* xI[1]) .+ (SjKj0Re .* xR .- SjKj0Im .* xI)
    Lcx0xt[div(end,2)+1:end] .= pre_fac*(vec(KimT[1,:]) .+ vec(KimT[:,1])) .+ (SjKTRe .* xI[1] .+ SjKTIm .* xR[1]) .+ (SjKj0Re .* xI .+ SjKj0Im .* xR)
    return [Lcx...,Lcx2...,Lcx0xt...]
end



function Lc_x_x2_x0xt_1D(u,H,m,λ,a,a_m1,gp1,gm1,κ)

    KRe,KIm = getK(H)

    Lcx = zeros(size(u))
    Lcx2 = zeros(size(u))
    Lcx0xt = zeros(size(u))

    xR =  @view u[1:div(end,2),:]
    xI =  @view u[div(end,2)+1:end,:]
    
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

    #A = im*pre_fac*( 
    #            (x .- x[gp1]) ./ a_m1 + (x .- x[gm1]) ./ a
    #            .- (a .+ a_m1)/2 .* (m .* x .+ (λ/6) .* x.^3)
    #    )
    #ARe = real(A)
    #AIm = imag(A)

    SjKTRe = KRe*ARe .- KIm*AIm
    SjKTIm = KRe*AIm .+ KIm*ARe
    
    #mul!(_B,K,[ARe ; AIm])
    #_B = _A#transpose(_B)

    #kReT = (p[1:div(end,2),:]*transpose(p[1:div(end,2),:]) 
    #            - p[div(end,2)+1:end,:]*transpose(p[div(end,2)+1:end,:]))
    #kImT = (p[1:div(end,2),:]*transpose(p[div(end,2)+1:end,:]) 
    #            + p[div(end,2)+1:end,:]*transpose(p[1:div(end,2),:]))
    
    #ARe = @view _A[1:div(end,2)]
    #AIm = @view _A[div(end,2)+1:end]
    #SjKTRe = vec(transpose(ARe)*KreT .- transpose(AIm)*KimT)
    #SjKTRe = @view _B[1:div(end,2)]#ARe #vec(transpose(ARe)*kReT .- transpose(AIm)*kImT)
    #SjKTIm = vec(transpose(AIm)*KreT .+ transpose(ARe)*KimT)
    #SjKTIm = @view _B[div(end,2) + 1:end]#AIm #vec(transpose(AIm)*kReT .+ transpose(ARe)*kImT)

    Lcx[1:div(end,2)]     .= SjKTRe  #vec((transpose(ARe)*kReT .- transpose(AIm)*kImT))
    Lcx[div(end,2)+1:end] .= SjKTIm #vec((transpose(AIm)*kReT .+ transpose(ARe)*kImT))
    

    
    
    Lcx2[1:div(end,2)]     .= pre_fac*2*diag(KreT) .+ 2*(SjKTRe .* xR .- SjKTIm .* xI)
    Lcx2[div(end,2)+1:end] .= pre_fac*2*diag(KimT) .+ 2*(SjKTRe .* xI .+ SjKTIm .* xR)
    
    Lcx2[1:1]     .= sum(pre_fac*2*diag(KreT) .+ 2*(SjKTRe .* xR .- SjKTIm .* xI))
    Lcx2[div(end,2)+1:div(end,2)+1] .= sum(pre_fac*2*diag(KimT) .+ 2*(SjKTRe .* xI .+ SjKTIm .* xR))
    

    
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







function getBoundaryTerms(KP::KernelProblem;Ys=nothing,Xs=nothing,kwargs...)

    @unpack model, kernel = KP

    if isnothing(Ys)
        Ys = collect(0:0.1:8)
    end

    L_CO,L2_CO = getBoundaryTermsObservables(KP;kwargs...)
    return BoundaryTerms(model,Ys,Xs,L_CO,L2_CO)
end






function calcBoundaryTerms(sol,BT::BoundaryTerms{MType,YT,L,L2};T=Float64, witherror=true,box=false) where {MType <: Model,YT,L,L2}
    @unpack Ys, Xs = BT
    NTr = length(sol)

    realDir = !isnothing(Xs)
    d = realDir ? 4 : 3

    NObs = 6*BT.model.contour.t_steps

    if realDir
        LOs = zeros(T,NObs,length(Ys),length(Xs),NTr)
        L2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys),length(Xs),NTr)
    else
        LOs = zeros(T,NObs,length(Ys),NTr)
        L2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys),NTr)
    end

    Threads.@threads for tr in 1:NTr
    #for tr in 1:NTr

        N = size(sol[tr])[2]
        trVs = zeros(T,NObs,N)
        trV2s = (L2 == Nothing) ? nothing : zeros(T,NObs,N)
        
        @inbounds @simd for i in 1:N
            tri = @view sol[tr][:,i]
            trVs[:,i] = BT.L_CO(tri)

            if L2 != Nothing
                trV2s[:,i] = BT.L2_CO(tri)
            end
        end
        

        ### TODO: This needs to be abstracted away to the specific model
        if length(sol[tr][:,1]) > 2
            if realDir || box
                XX = @view sol[tr][1:div(end,2),:]
                YY = @view sol[tr][div(end,2)+1:end,:]
                imX = [sum(abs.(XX.^2 .- YY.^2),dims=1) ; sum(abs.(2 .* XX .* YY),dims=1)]
                #imX = [maximum(abs.(XX),dims=1) ; maximum(abs.(YY),dims=1)]
            else
                imX = maximum(abs.(@view sol[tr][div(end,2)+1:end,:]),dims=1)
            end
        else
            #imX = maximum(abs.(sol[tr]),dims=1)
            if realDir || box
                imX = abs.(sol[tr])
            else
                imX = abs.(@view sol[tr][2,:])
            end
        end

        if realDir
            trLOs = zeros(T,NObs,length(Ys),length(Xs))
            trL2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys),length(Xs))
        else
            trLOs = zeros(T,NObs,length(Ys))
            trL2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys))
        end
        for i in eachindex(Ys)
            if realDir
                for j in eachindex(Xs)
                    Hinx = @. (imX[2,:] .<= Ys[i]) .& (imX[1,:] .<= Xs[j])
                    H = @view trVs[:,Hinx]
                    trLOs[:,i,j] = sum(H,dims=2)/N
                    if L2 != Nothing
                        H2 = @view trV2s[:,Hinx]
                        trL2Os[:,i,j] = sum(H2,dims=2)/N
                    end
                end
            elseif box
                Hinx = @. (imX[2,:] .<= Ys[i]) .& (imX[1,:] .<= Ys[i])
                H = @view trVs[:,Hinx]
                trLOs[:,i] = sum(H,dims=2)/N
                if L2 != Nothing
                    H2 = @view trV2s[:,Hinx]
                    trL2Os[:,i] = sum(H2,dims=2)/N
                end
            else 
                Hinx = @. (imX[:] .<= Ys[i])
                H = @view trVs[:,Hinx]
                trLOs[:,i] = sum(H,dims=2)/N
                if L2 != Nothing
                    H2 = @view trV2s[:,imX .<= Ys[i]]
                    trL2Os[:,i] = sum(H2,dims=2)/N
                end
            end
        end
        LOs[map((i) -> :,1:(d-1))...,tr] .= trLOs
        if L2 != Nothing
            L2Os[map((i) -> :,1:(d-1))...,tr] .= trL2Os
        end
    end
    if L2 == Nothing
        if witherror
            return dropdims(mean(LOs,dims=d),dims=d) .± dropdims(std(LOs,dims=d),dims=d) ./ sqrt(NTr), 
                    nothing
        else
            return dropdims(mean(LOs,dims=d),dims=d),# .± dropdims(std(LOs,dims=d),dims=d) ./ sqrt(NTr), 
                    nothing
        end
    else
        return dropdims(mean(LOs,dims=d),dims=d) .± dropdims(std(LOs,dims=d),dims=d) ./ sqrt(NTr),
               dropdims(mean(L2Os,dims=d),dims=d) .± dropdims(std(L2Os,dims=d),dims=d) ./ sqrt(NTr)
    end
end

function calcBoundaryTermsCorrections(sol,BT::BoundaryTerms,lims::Tuple{T, T},lims2::Tuple{T, T}) where {T <: Real}
    isnothing(BT.L2_CO) && return 0.
    L_CO, L2_CO = calcBoundaryTerms(sol,BT)

    # Interpolate the boundary value
    Yinx = findall((x) -> lims[1] <= x <= lims[2], BT.Ys)
    Yinx2 = findall((x) -> lims2[1] <= x <= lims2[2], BT.Ys)
    B1 = weightedmean(L_CO[Yinx])
    B2 = weightedmean(L2_CO[Yinx2])
    return B1, B2
end


