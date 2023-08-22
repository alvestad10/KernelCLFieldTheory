export setKernel!, MatrixKernel, SparseMatrixKernel

abstract type Kernel end

abstract type ConstantKernel <: Kernel end


function getK(H)
    _H = H[1:div(end,2),:] .+ im*H[div(end,2)+1:end,:]
    _K = _H*transpose(_H)
    KRe = real(_K)
    KIm = imag(_K)
    return KRe, KIm
end

function getK(kernel::T) where {T <: ConstantKernel} 
    return getK(kernel.H)
end


"""
Constant kernel with the entries of the complex matrix H as the parameters.
We get the ``K`` by built in squaring ``K=H^T*H``
"""
mutable struct MatrixKernel{T} <: ConstantKernel
    H::Matrix{T}
end

function MatrixKernel(M::ScalarField{0})
    @unpack n_steps = M 
    @unpack t_steps = M.contour 

    _HRe = zeros(t_steps,t_steps)
    for i in 1:t_steps
            _HRe[i,i] = 1.
    end


    H = vcat(reshape(_HRe,t_steps,t_steps),
             zeros(t_steps,t_steps))
    
    return MatrixKernel(H)
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

function MatrixKernel(M::ScalarField{2})
    @unpack n_steps = M 
    @unpack t_steps = M.contour 

    _HRe = zeros(t_steps,n_steps,n_steps,t_steps,n_steps,n_steps)
    for i in 1:t_steps
        for j in 1:n_steps
            for k in 1:n_steps
                _HRe[i,j,k,i,j,k] = 1.
            end
        end
    end


    H = vcat(reshape(_HRe,t_steps*n_steps^2,t_steps*n_steps^2),
             zeros(t_steps*n_steps^2,t_steps*n_steps^2))
    
    return MatrixKernel(H)
end

function MatrixKernel(M::ScalarField{3})
    @unpack n_steps = M 
    @unpack t_steps = M.contour 

    _HRe = zeros(t_steps,n_steps,n_steps,n_steps,t_steps,n_steps,n_steps,n_steps)
    for i in 1:t_steps
        for j in 1:n_steps
            for k in 1:n_steps
                for l in 1:n_steps
                    _HRe[i,j,k,l,i,j,k,l] = 1.
            
                end    
            end
        end
    end


    H = vcat(reshape(_HRe,t_steps*n_steps^3,t_steps*n_steps^3),
             zeros(t_steps*n_steps^3,t_steps*n_steps^3))
    
    return MatrixKernel(H)
end


"""
Constant kernel with the entries of the complex matrix H as the parameters.
We get the ``K`` by built in squaring ``K=H^T*H``
"""
mutable struct _SparseMatrixKernel <: ConstantKernel
    H #::Array{T, N}
    convDims
end

# function getK(H::Sparse{T, N}) where {T <: Real, N}
#     _H = H[1:div(end,2),:] .+ im*H[div(end,2)+1:end,:]
#     _K = _H*transpose(_H)
#     KRe = real(_K)
#     KIm = imag(_K)
#     return KRe, KIm
# end

# function getK(kernel::_SparseMatrixKernel)
#     return getK(kernel.H)
# end

const spzero_val = 1e-24

function SparseMatrixKernel(M::ScalarField{0},convDims)
    @unpack n_steps = M 
    @unpack t_steps = M.contour 

    _HRe = spzeros(t_steps,t_steps)
    _HIm = spzeros(t_steps,t_steps)
    for i in 1:t_steps
        _HRe[i,i] = 1.
        _HIm[i,i] = spzero_val
        for j in 1:convDims 
            _HRe[i,mod1(i+j,t_steps)] = spzero_val
            _HRe[i,mod1(i-j,t_steps)] = spzero_val
            _HIm[i,mod1(i+j,t_steps)] = spzero_val
            _HIm[i,mod1(i-j,t_steps)] = spzero_val
        end
    end


    H = vcat(_HRe,
             _HIm)
    
    return _SparseMatrixKernel(H,convDims)
end


function SparseMatrixKernel(M::ScalarField{1}, convDims)
    @unpack n_steps = M 
    @unpack t_steps = M.contour 

    _HRe = spzeros(t_steps*n_steps,t_steps*n_steps)
    _HIm = spzeros(t_steps*n_steps,t_steps*n_steps)
    for i in 1:n_steps
        t_range = (i-1)*t_steps .+ (1:t_steps)

        _HRe[t_range,t_range] .= spzero_val .+ Diagonal(ones(t_steps))
        _HIm[t_range,t_range] .= spzero_val
        for j in 1:convDims 
            _HRe[t_range,mod(i+j-1,n_steps)*t_steps .+ (1:t_steps)] .= spzero_val
            _HRe[t_range,mod(i-j-1,n_steps)*t_steps .+ (1:t_steps)] .= spzero_val
            _HIm[t_range,mod(i+j-1,n_steps)*t_steps .+ (1:t_steps)] .= spzero_val
            _HIm[t_range,mod(i-j-1,n_steps)*t_steps .+ (1:t_steps)] .= spzero_val
        end
    end


    H = vcat(_HRe,
             _HIm)
    
    return _SparseMatrixKernel(H,convDims)
end


function SparseMatrixKernel(M::ScalarField{2})
    @unpack n_steps = M 
    @unpack t_steps = M.contour 

    _HRe = spzeros(t_steps*n_steps*n_steps,t_steps*n_steps*n_steps)
    _HIm = spzeros(t_steps*n_steps*n_steps,t_steps*n_steps*n_steps)
    for i in 1:n_steps
        t_range = (i-1)*t_steps*n_steps .+ (1:t_steps)

        _HRe[t_range,t_range] .= spzero_val .+ Diagonal(ones(t_steps))
        _HIm[t_range,t_range] .= spzero_val
        for j in 1:convDims 
            _HRe[t_range,mod(i+j-1,n_steps)*t_steps*n_steps .+ (1:t_steps)] .= spzero_val
            _HRe[t_range,mod(i-j-1,n_steps)*t_steps*n_steps .+ (1:t_steps)] .= spzero_val
            _HIm[t_range,mod(i+j-1,n_steps)*t_steps*n_steps .+ (1:t_steps)] .= spzero_val
            _HIm[t_range,mod(i-j-1,n_steps)*t_steps*n_steps .+ (1:t_steps)] .= spzero_val
        end
    end


    H = vcat(reshape(_HRe,t_steps*n_steps^2,t_steps*n_steps^2),
             zeros(t_steps*n_steps^2,t_steps*n_steps^2))
    
    return MatrixKernel(H)
end