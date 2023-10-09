export KernelProblem

"""
    Contains the information needed to run a simulation
    
    model: The model to be simulated
    kernel: Kernel used in the simulation
    a: Drift term function
    b: noise term function (without the WienerProcess)
    y: Known solutions to the model 
"""
struct KernelProblem{MType<:Model, KType<:Kernel,AType,BType,yType}
    model::MType
    kernel::KType
    a::AType
    b::BType
    y::yType
    κ::Float64
end


"""
    Construct kernel problem based on model and kernel
"""
function KernelProblem(model::Model;kernel=MatrixKernel(model), κ = 0.0)
    
    a_func!, b_func! =  get_ab(model,kernel; κ = κ)

    y = getSolutions(model)

    return KernelProblem(model,kernel,a_func!,b_func!,y,κ)
end

Base.copy(KP::KernelProblem) = KernelProblem(KP.model;kernel=KP.kernel)

"""
    Updating the kernel problem after update in the kernel parameters
"""
function updateProblem(KP::KernelProblem; κ=nothing)
    if isnothing(κ)
        κ = KP.κ
    end
    return KernelCLFieldTheory.KernelProblem(KP.model, kernel=KP.kernel, κ=κ)
end