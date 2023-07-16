using KernelCLFieldTheory
using Plots
using Zygote
using Measurements
using Statistics
using JLD2

DIR = "PaperPlots/NewCodeTests/Data/RT_10_a_01/"
if !isdir(dirname(obs_path))
    mkdir(dirname(obs_path))
end

M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.0,β=0.4,at=0.1,n_steps=16,as=0.1,
                        Δβ = 0.5                
)

KP = KernelCLFieldTheory.KernelProblem(M)
RS = RunSetup(NTr=1, tspan=1000, saveat=0.01, dt=1e-4, dtmax=1e-3, adaptive=true)

@time sol = run_simulation(KP,RS);

plotSKContour(KP,sol)

obs = KernelCLFieldTheory.calc_obs(KP,sol)

mean(obs[1] .± obs[2])
mean(obs[3] .± obs[4])
mean(obs[5] .± obs[6])
mean(obs[7] .± obs[8])


obs_path = joinpath(DIR,"obs.jld2")
jldsave(obs_path; obs)

KP_path = joinpath(DIR,"KP.jld2")
jldsave(KP_path; KP)



