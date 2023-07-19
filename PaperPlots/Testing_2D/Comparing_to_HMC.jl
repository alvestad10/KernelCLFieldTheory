using KernelCLFieldTheory
using Plots
using Measurements
using Statistics
using JLD2
using LinearAlgebra


M = ScalarField_from_at(D=2,m=1.0,λ=1.0,RT=0.0,β=0.4,at=0.1,n_steps=16,as=0.1,
                        #Δβ = 0.5                
                        #ΔE = 0.05              
)

KP = KernelCLFieldTheory.KernelProblem(M);
RS = RunSetup(NTr=1, tspan=30000, saveat=0.1, dt=1e-4, dtmax=1e-3, adaptive=true, 
                    scheme = KernelCLFieldTheory.LambaEM())

@time sol = run_simulation(KP,RS);

#_sol = sol

plotSKContour(KP,sol)
plotFWContour(KP,sol)

phi2Re,phi2Re_err = KernelCLFieldTheory.calc_obs(KP,sol)[5:6]
mean(phi2Re .± phi2Re_err)

begin
    max_inx = length(sol[1])
    arr = zeros(typeof(1.0 ± 1.0), ceil(Int,max_inx/10000))
    for (i,k) in enumerate(collect(1:10000:max_inx))
        phi2Re,phi2Re_err = KernelCLFieldTheory.calc_obs(KP,sol;max_inx=k)[5:6]
        arr[i] = mean(phi2Re .± phi2Re_err)
    end
    plot(arr)
end



arr_01 = arr
arr_02 = arr


begin
    plot(arr_01, label="a=0.1")
    plot(arr_02, label="a=0.2")
end

begin
    scatter([0.2],[mean([0.5405 ± 0.0041, 0.5414 ± 0.0041])], label="a=0.2")
    scatter!([0.198],[0.5364 ± 0.003], label="a=0.2")
    scatter!([0.195],[0.537 ± 0.016], label="a=0.2")
    scatter!([0.1],[0.4403 ± 0.0017], label="a=0.1")
    scatter!([0.098],[0.426 ± 0.027], label="a=0.1")
    scatter!([0.096],[0.4072 ± 0.0016], label="a=0.1")
    scatter!([0.05],[0.2443 ± 0.002], label="a=0.05")
end
