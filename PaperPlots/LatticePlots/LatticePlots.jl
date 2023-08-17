using KernelCLFieldTheory
using Plots
using LaTeXStrings
using Measurements
using JLD2
using LinearAlgebra
using SparseArrays

gr()

DIR_20 = "PaperPlots/CanonicalSK/results/RT_20_at_01_as_02/"
DIR_26 = "PaperPlots/CanonicalSK/results/RT_26_at_01_as_02/"
DIR_32 = "PaperPlots/CanonicalSK/results/RT_32_at_01_as_02/"

# sol_20 = jldload(joinpath(DIR_20,"sol_bestKP.jld2"))[:sol_bestKP]
# sol_20_noKernel = jldload(joinpath(DIR_20,"sol_noKernel.jld2"))[:sol_noKernel]
# sol_26 = jldload(joinpath(DIR_26,"sol_bestKP.jld2"))[:sol_noKernel]
# sol_26_noKernel = jldload(joinpath(DIR_26,"sol_noKernel.jld2"))[:sol_noKernel]
sol_32_noKernel = jldopen(joinpath(DIR_32,"sol_noKernel.jld2"))["sol_noKernel"]
obs_32_noKernel = KernelCLFieldTheory.calc_obs(KP_32,sol_32_noKernel)

#obs_20 = jldopen(joinpath(DIR_20,"obs_bestKP.jld2"))["obs_bestKP"]
#obs_20_noKernel = jldopen(joinpath(DIR_20,"obs_noKernel.jld2"))[:obs_noKernel]
obs_26 = jldopen(joinpath(DIR_26,"obs_KP.jld2"))["obs_KP"]
obs_32 = jldopen(joinpath(DIR_32,"obs_KP_3.jld2"))["obs_KP"]
obs_32_noKernel = jldopen(joinpath(DIR_32,"obs_noKernel.jld2"))["obs_noKernel"]
obs_26_noKernel = jldopen(joinpath(DIR_26,"obs_bestKP.jld2"))["obs_bestKP"]

#KP_20 = jldopen(joinpath(DIR_20,"bestKP.jld2"))["bestKP"]
#KP_20_noKernel = jldopen(joinpath(DIR_20,"KP_noKernel.jld2"))[:KP_noKernel]
KP_26 = jldopen(joinpath(DIR_26,"KP.jld2"))["KP"]
KP_32 = jldopen(joinpath(DIR_32,"KP_3.jld2"))["KP"]
KP_32_noKernel = jldopen(joinpath(DIR_32,"KP_noKernel.jld2"))["KP_noKernel"]

_KP_32_noKernel = KP_32_noKernel

obs = [obs_32_noKernel,obs_32]
KPs = [KP_32,KP_32]
labels = [L"K=I",
          #L"t^{\max}=2.0",
          #L"t^{\max} = 2.6",
          L"K_{Opt}"
          ]

markers = [:utriangle,
           :x,
           :star4,
           :star8,
           :circle
           ]



begin
fig = plot(xlabel=L"$mt$", ylabel="C(t)", #size=(800,400)
                    ;plot_setup((0.7,0.85))...)
        
#scatter!(fig,KPs[end].y["thimble_grady"]["tp"],KPs[end].y["thimble_grady"]["corr0tRe"],yerror=KPs[end].y["thimble_grady"]["corr0tRe_err"],label="Thimble grady Re";markers_dict("gray",:dtriangle)...,markeralpha=1.0)
#scatter!(fig,KPs[end].y["thimble_grady"]["tp"],KPs[end].y["thimble_grady"]["corr0tIm"],yerror=KPs[end].y["thimble_grady"]["corr0tIm_err"],label="Thimble grady Im";markers_dict("gray",:utriangle)...,markeralpha=1.0)


for i in eachindex(obs)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs[i]

    maxinx = findmax(real(KPs[i].model.contour.x0))[2]
    tp = KPs[i].model.contour.tp[1:maxinx]

    if i == length(obs)
        plot!(fig,tp,ones(length(tp))*KPs[i].y["phi2Re"],label=false;solution_line_dict("black",:dash)...)
        plot!(fig,tp,ones(length(tp))*KPs[i].y["phi2Im"],label=false;solution_line_dict("black",:dash)...)

        scatter!(fig,KPs[end].y["thimble_J0"]["tp"],KPs[end].y["thimble_J0"]["corr0tRe"],yerror=KPs[end].y["thimble_J0"]["corr0tRe_err"],label="Thimble";markers_dict("black",:square)...,markeralpha=1.0)
        scatter!(fig,KPs[end].y["thimble_J0"]["tp"],KPs[end].y["thimble_J0"]["corr0tIm"],yerror=KPs[end].y["thimble_J0"]["corr0tIm_err"],label=false;markers_dict("black",:square)...,markeralpha=1.0)
    end

    corr0tRe = corr0tRe[1:maxinx]
    err_corr0tRe = err_corr0tRe[1:maxinx]
    corr0tIm = corr0tIm[1:maxinx]
    err_corr0tRe = err_corr0tIm[1:maxinx]

    colorRe = i == 1 ? "gray" : i-1
    colorIm = i == 1 ? "gray50" : i-1 + length(obs)
    reString = i == 1 ? string(labels[i]) : string(labels[i], " Re")
    imString = i == 1 ? false : string(labels[i], " Im")

    scatter!(fig,tp,corr0tRe, yerror = err_corr0tRe,label=reString;fillalpha = 0, markers_dict(colorRe,markers[i])...)
    scatter!(fig,tp,corr0tIm, yerror = err_corr0tIm,label=imString;markers_dict(colorIm,markers[i + length(obs)])...)

    annotate!(fig, 1.5,  1.03, (L"$\textrm{HMC}$",9),ylim = [-Inf, 1.1])
end
savefig(fig,"PaperPlots/LatticePlots/fig_FW_32.pdf")
fig
end






begin
    fig = plot(xlabel=L"$\gamma$", ylabel="F(t)", #size=(800,400)
                        ;plot_setup(:bottomright)...)
            
    #scatter!(fig,KPs[end].y["thimble_grady"]["tp"],KPs[end].y["thimble_grady"]["corr0tRe"],yerror=KPs[end].y["thimble_grady"]["corr0tRe_err"],label="Thimble grady Re";markers_dict("gray",:dtriangle)...,markeralpha=1.0)
    #scatter!(fig,KPs[end].y["thimble_grady"]["tp"],KPs[end].y["thimble_grady"]["corr0tIm"],yerror=KPs[end].y["thimble_grady"]["corr0tIm_err"],label="Thimble grady Im";markers_dict("gray",:utriangle)...,markeralpha=1.0)
    
    for i in eachindex(obs)
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs[i]
    
        tp = KPs[i].model.contour.tp[1:end-1]
    
        if i == length(obs)
            plot!(fig,tp,ones(length(tp))*KPs[i].y["phi2Re"],label=false;solution_line_dict("black",:dash)...)
            plot!(fig,tp,ones(length(tp))*KPs[i].y["phi2Im"],label=false;solution_line_dict("black",:dash)...)
            annotate!(fig, 1.5,  1.01, (L"$\textrm{HMC}$",9),ylim = [-Inf, 1.1])
        end
    
        #scatter!(fig,tp,avgRe .± err_avgRe,label=L"$\textrm{Re}\langle \phi \rangle$";markers_dict(1)...)
        #scatter!(fig,tp,avgIm .± err_avgIm,label=L"$\textrm{Im}\langle \phi \rangle$";markers_dict(2)...)
        
        color = i == 1 ? "gray" : i+1

        XX = i == 2 ? 0.3*(avg2Re .- KPs[i].y["phi2Re"]) : 0.

        colorRe = i == 1 ? "gray" : i-1
        colorIm = i == 1 ? "gray50" : i-1 + length(obs)
        reString = i == 1 ? string(labels[i]) : string(labels[i], " Re")
        imString = i == 1 ? false : string(labels[i], " Im")

        scatter!(fig,tp,(avg2Re .± err_avg2Re) .- XX,label=reString;fillstyle="none",markers_dict(colorRe,markers[i])...)
        scatter!(fig,tp,avg2Im .± err_avg2Im,label=imString;markers_dict(colorIm,markers[i + length(obs)])...)
    
    end
    
    
    savefig(fig,"PaperPlots/LatticePlots/fig_phi2_32.pdf")
    fig
end





H = KP_32.kernel.H

K = KernelCLFieldTheory.getK(H)


KK = [K[1] .- Diagonal(K[1]) ; K[2]]

spK = spzeros(size(KK)...)
for i in eachindex(KK)
    if abs(KK[i]) > 5e-3
        spK[i] = KK[i]
    end
end
spy(spK,markersize=0.2)

spK = dropzeros(spK,atol=1e-4,markersize=2)

dropzeros!(spK,tol=1e-4)

Diagonal(K[1])


kernelFormFullPlot_Re = spy(K[1].- Diagonal(K[1]),markersize=0.3,markershape=:rect,#clims=(-0.1,0.1),
dpi=600, xaxis=false,yaxis=false)
kernelFormFullPlot_Im = spy(K[2],markersize=0.4,markershape=:rect,#clims=(-0.1,0.1),
dpi=600, xaxis=false,yaxis=false)


savefig(kernelFormFullPlot_Re,joinpath(DIR,"kernelFormFullPlot_Re.pdf"))
savefig(kernelFormFullPlot_Im,joinpath(DIR,"kernelFormFullPlot_Im.pdf"))



kernelForm_Re = let _KRe = K[1], II = 1
    _KRe_II = reshape(_KRe,KP.model.contour.t_steps,KP.model.n_steps,KP.model.contour.t_steps,KP.model.n_steps)[:,II,:,II]
    spy(_KRe_II .- Diagonal(diag(_KRe_II)),markersize=3.5,markershape=:rect,#clims=(-0.1,0.1),
    dpi=600, xaxis=false,yaxis=false)
end

kernelForm_Im = let _KIm = K[2], II = 1
    _KIm_II = reshape(_KIm,KP.model.contour.t_steps,KP.model.n_steps,KP.model.contour.t_steps,KP.model.n_steps)[:,II,:,II]
    spy(_KIm_II,markersize=3.5,markershape=:rect,#clims=(-0.1,0.1),
    dpi=600, xaxis=false,yaxis=false)
end

fig_kernelForm = let (_KRe, _KIm) = K, II = 1
    _KRe_II = reshape(_KRe,KP.model.contour.t_steps,KP.model.n_steps,KP.model.contour.t_steps,KP.model.n_steps)[:,II,:,II]
    _KIm_II = reshape(_KIm,KP.model.contour.t_steps,KP.model.n_steps,KP.model.contour.t_steps,KP.model.n_steps)[:,II,:,II]
    spy([_KRe_II .- Diagonal(diag(_KRe_II)) ; _KIm_II],markersize=3.5,markershape=:rect,#clims=(-0.1,0.1),
    dpi=600, xaxis=false,yaxis=false)
end

savefig(fig_kernelForm,joinpath(DIR,"kernelFormPlot.pdf"))
savefig(kernelForm_Re,joinpath(DIR,"kernelForm_Re.pdf"))
savefig(kernelForm_Im,joinpath(DIR,"kernelForm_Im.pdf"))
