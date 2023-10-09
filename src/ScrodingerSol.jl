
export getScroSol

#################################################################
#################################################################


######
# SETTINGS
######
const NSCRO = 128
const TSCRO = BigFloat
const DIR = "./ScroSols"

function setupScroSol()
    if !isdir(DIR)
        mkdir(DIR)
    end
end

setupScroSol()
######



function getH(R, M::ScalarField{0})

    @unpack m, λ = M

    i = R[1]-1
    j = R[2]-1
    if i == j + 4
        return √prod(j+1:j+4) * λ/(96m^2)
    elseif i == j + 2
         return √prod(j+1:j+2) * (2j+3)λ/(48m^2)
    elseif i == j
        return (2j^2 + 2j + 1) * λ/(32m^2) + (j+0.5)m
    elseif i == j-2
        return √prod(j-1:j) * (2j-1)λ/(48m^2)
    elseif i == j-4
        return √prod(j-3:j) * λ/(96m^2)
    end
    
    return 0
end


function getXTrans(R, M::ScalarField{0})
    @unpack m = M
    
    i = R[1]-1
    j = R[2]-1

    if i == j - 1
        return (1/sqrt(2m)) * √j
    elseif i == j+1
        return (1/sqrt(2m)) * √(j+1)
    end
    
    return 0
end

function fill_Hamiltonian(M::ScalarField{0},dim)
    # Allocating the Hamiltonian matrix
    H = zeros(TSCRO,dim,dim);
    #H = spzeros(dim,dim)

    # Builder the HAmiltonian matrix
    R = CartesianIndices(H)
    Ifirst, Ilast = first(R), last(R)
    I1 = CartesianIndex(0, 1)
    for I in diag(R) 
        for J in max(Ifirst, I-4I1):min(Ilast, I+4I1)
            H[J] = getH(J,M)
        end
    end
    return H
end



struct ScrodingerProblem{HType <: AbstractArray, ZType, TTimeEvol,TEvecs,TBoltz,TXTrans}
    dim::Integer
    M::ScalarField{0}
    H::HType
    Z::ZType
    timeEvol::TTimeEvol
    evecs::TEvecs
    boltz::TBoltz
    xtrans::TXTrans
end

function ScrodingerProblem(M::ScalarField{0},dim,T::Type)

    # DEfining the model
    H = fill_Hamiltonian(M,dim)

    evals, evecs = eigen(H)
    p = sortperm(real(evals), rev=true)
    evals = map(pp -> T(evals[pp]),p);
    #evecs = map(xx -> round(xx,digits=40),evecs[:,p]);
    evecs = map(xx -> T(xx),evecs[:,p]);

    # Time ovolution operator: Exp[iHt] = Exp[iEt]
    timeEvol(t) = Diagonal(map(eval -> Complex{T}(exp( im*t*eval)), evals))
    
    # Boltzman factor Exp[-\[Beta]E]
    boltz = Diagonal(map(eval -> T(exp(-M.contour.β*eval)), evals));

    x = spzeros(T,dim,dim);
    R = CartesianIndices(x)
    Ifirst, Ilast = first(R), last(R)
    I1 = CartesianIndex(0, 1)
    for I in diag(R) 
        for J in max(Ifirst, I-I1):min(Ilast, I+I1)
            x[J] = getXTrans(J,M)
        end
    end
    
    xtrans = transpose(evecs) * x * evecs;

    Z = sum(map(eval -> exp(-M.contour.β*eval), evals));

    return ScrodingerProblem(dim,M,H,Z,timeEvol,evecs,boltz,xtrans)
end

function RTCorr0t(t,prob::ScrodingerProblem)
    @unpack Z, evecs, boltz, timeEvol, xtrans = prob
    tmp = evecs * boltz * timeEvol(t) * xtrans * timeEvol(-t) * xtrans * transpose(evecs) 
    (1/Z) * tr(tmp)
end

function RTCorrAvg2(t,prob::ScrodingerProblem)
    @unpack Z, evecs, boltz, timeEvol, xtrans = prob
    tmp = evecs * boltz * timeEvol(t) * xtrans * xtrans * timeEvol(-t)* transpose(evecs)
    (1/Z) * tr(tmp)
end

function RTCorrAvg(t,prob::ScrodingerProblem)
    @unpack Z, evecs, boltz, timeEvol, xtrans = prob
    tmp = evecs * boltz * timeEvol(t) * xtrans* timeEvol(-t) * transpose(evecs)
    (1/Z) * tr(tmp)
end


function getScroSolNumberMax()
    files = readdir(DIR)
    NMax = 0
    for file in files
        if parse(Int,split(file,"_")[end]) > NMax
            NMax = parse(Int,split(file,"_")[end])
        end
    end 

    return NMax
end

function isSameContour(contour1,contour2)
    !(contour1.t_steps == contour2.t_steps) && return false
    return all(contour1.x0 .== contour2.x0)
end

function isSameModel(model1::ScalarField{0},model2::ScalarField{0})
    contour1 = model1.contour
    contour2 = model2.contour
    return all([model1.m == model2.m,
                model1.λ == model2.λ,
                isSameContour(contour1,contour2)])
end

function checkIfAlreadyCalculated(model::ScalarField{0})

    @unpack RT,β,t_steps = model.contour

    files = readdir(DIR)

    for file in files
        if string(split(file,"_")[1]) !== "Lookup"
            continue
        end

        lo = load(joinpath(DIR,file))
        file_model = lo[":model"]
        

        if isSameModel(model,file_model)
            return parse(Int,split(file,"_")[end])
        end
    end

    return 0
end

function getSolutions(model::ScalarField{0})
    @unpack contour = model
    
    scroNumber = checkIfAlreadyCalculated(model)

    if scroNumber == 0
        prob = ScrodingerProblem(model,NSCRO,TSCRO)

        corr0t = zeros(Complex{Float64},length(contour.a))
        @Threads.threads for (i,t) in collect(enumerate(contour.x0[1:end-1]))
            sol = RTCorr0t(t,prob)
            corr0t[i] = ComplexF64(sol)
        end

        y_x = ComplexF64(RTCorrAvg(0.,prob)) # .* ones(length(contour.a)))
        y_x2 = ComplexF64(corr0t[1]) # .* ones(length(contour.a)))

        scroSol = Dict("phiRe" => real(y_x), "phiIm" => imag(y_x),
                       "phi2Re" => real(y_x2), "phi2Im" => imag(y_x2),
                       "corr0tRe" => real(corr0t),
                       "corr0tIm" => imag(corr0t),)
        ScroNumber = getScroSolNumberMax()+1

        filename = string("ScroSol_",ScroNumber)
        filename_lookup = string("Lookup_",ScroNumber)
        @save joinpath(DIR,filename) :scroSol=scroSol
        @save joinpath(DIR,filename_lookup) :model=model

        return scroSol
    else
        lo = load(joinpath(DIR,string("ScroSol_",scroNumber)))
        return  lo[":scroSol"]
    end    
    
end

