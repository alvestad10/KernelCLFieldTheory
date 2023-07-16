export setKernel!

abstract type Kernel end

abstract type ConstantKernel <: Kernel end

"""
Constant kernel with the entries of the complex matrix H as the parameters.
We get the ``K`` by built in squaring ``K=H^2``
"""
mutable struct MatrixKernel{T} <: ConstantKernel
    H::Matrix{T}
end

function getK(H::Matrix{T}) where {T <: Real}
    _H = H[1:div(end,2),:] .+ im*H[div(end,2)+1:end,:]
    K = _H*transpose(_H)
    return K
end

function getK(kernel::MatrixKernel)
    _K = getK(kernel.H)
    KRe = real(_K)
    KIm = imag(_K)
    return KRe, KIm
end

function MatrixKernel(M::ScalarField{1})
    @unpack n_steps = M 
    @unpack t_steps = M.contour 

    _HRe = zeros(t_steps,n_steps,t_steps,n_steps)
    for i in 1:t_steps
        for j in 1:n_steps
            _HRe[i,j,i,j] = 1.
        end
    end


    H = vcat(reshape(_HRe,t_steps*n_steps,t_steps*n_steps),
             zeros(t_steps*n_steps,t_steps*n_steps))
    
    return MatrixKernel(H)
end



