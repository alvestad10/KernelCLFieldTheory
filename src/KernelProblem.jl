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
end


"""
    Construct kernel problem based on model and kernel
"""
function KernelProblem(model::Model;kernel=MatrixKernel(model))
    
    a_func!, b_func! =  get_ab(model,kernel)

    y = getSolutions(model)

    return KernelProblem(model,kernel,a_func!,b_func!,y)
end

Base.copy(KP::KernelProblem) = KernelProblem(KP.model;kernel=KP.kernel)

"""
    Updating the kernel problem after update in the kernel parameters
"""
function updateProblem(KP::KernelProblem)
    @unpack model, kernel, a, b, y = KP

    updateKernel!(kernel.pK)
    a_func!, b_func! = get_ab(model,kernel)

    return KernelProblem(model, kernel, a_func!, b_func!, y) 
end