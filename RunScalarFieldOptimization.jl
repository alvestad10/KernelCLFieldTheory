using KernelCLFieldTheory
using Plots

M = ScalarField_from_at(D=1,m=1.0,λ=1.0,RT=1.0,β=0.4,at=0.1,n_steps=8,as=0.1)

KernelCLFieldTheory.get_initial_value(M)
KernelCLFieldTheory.get_noise_rate_prototype(M)
KernelCLFieldTheory.get_caches(M)

kernel = KernelCLFieldTheory.MatrixKernel(M)

y = getSolutions(M)

KernelCLFieldTheory.get_ab(M,kernel)

KP = KernelCLFieldTheory.KernelProblem(M)
RS = RunSetup(NTr=1, tspan=50, saveat=0.01, dt=1e-4, dtmax=1e-3, adaptive=true)

KP.kernel.H

sol = run_simulation(KP,RS)


obs = KernelCLFieldTheory.calc_obs(KP,sol)


if RS.NTr == 1
    avgRe, err_avgRe, avgIm, err_avgIm,avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs
else
    avgRe, err_avgRe, avgIm, err_avgIm,avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = KernelCLFieldTheory.calc_meanObs(obs)
end

begin
    fig = plot()
    plot!(fig,avgRe)
    plot!(fig,avgIm)
    plot!(fig,avg2Re)
    plot!(fig,avg2Im)
    plot!(fig,corr0tRe)
    plot!(fig,corr0tIm)
end