export plotSKContour

default(palette = palette(:default))

M_SIZE = 4.0
SOL_LW = 1.0

plot_setup() = plot_setup(:bottomright)

plot_setup(legend) = Dict(:legend     =>  legend,
                        :grid       => false,
                        :foreground_color_legend=> nothing,
                        :background_color_legend=> nothing,
                        #:bottom_margin => -2mm,
                        #:left_margin => -3mm,
                        :framestyle => :box,
                        :thickness_scaling => 1.3)

markers_dict(color,markershape) = Dict(
                        :color => color,
                        :markershape => markershape,
                        :markersize  => M_SIZE,
                        :markerstrokecolor => color)#,
                        #:lw => false)

markers_dict(color) = markers_dict(color,:square) 

solution_line_dict(color) = Dict(:color => color,:lw => SOL_LW)
solution_line_dict() = solution_line_dict("black")


function plotSKContour(KP::KernelProblem,sol;obs=nothing)
    
    if isnothing(obs)
        obs = calc_obs(KP,sol)
    end
    
    if sol isa SciMLBase.RODESolution || length(sol) == 1
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = obs
    elseif sol isa SciMLBase.EnsembleSolution
        avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(obs)
    end

    
    tp = KP.model.contour.tp[1:end-1]


    legend_outside = true

    if legend_outside
        fig = plot(xlabel=L"$\gamma$", size=(800,400)
                    ;plot_setup(:outerright)...)
    else
        fig = plot(xlabel=L"$\gamma$",#ylim=[-0.31,0.34]
                    ;plot_setup(:bottomright)...)
    end

    plot!(fig,tp,ones(length(tp))*KP.y["phi2Re"],label=L"$\textrm{HMC}$";solution_line_dict()...)
    plot!(fig,tp,ones(length(tp))*KP.y["phi2Im"],label=false;solution_line_dict()...)
    
    scatter!(fig,tp,avgRe .± err_avgRe,label=L"$\textrm{Re}\langle \phi \rangle$";markers_dict(1)...)
    scatter!(fig,tp,avgIm .± err_avgIm,label=L"$\textrm{Im}\langle \phi \rangle$";markers_dict(2)...)
    
    scatter!(fig,tp,avg2Re .± err_avg2Re,label=L"$\textrm{Re}\langle \phi^2 \rangle$";markers_dict(3,:square)...)
    scatter!(fig,tp,avg2Im .± err_avg2Im,label=L"$\textrm{Im}\langle \phi^2 \rangle$";markers_dict(4,:circle)...)

    if "corr0tRe" in keys(KP.y)
        plot!(fig,tp,KP.y["corr0tRe"],label=false;solution_line_dict()...)
        plot!(fig,tp,KP.y["corr0tIm"],label=false;solution_line_dict()...)
    end
    scatter!(fig,tp,corr0tRe, yerror = err_corr0tRe,label=L"$\textrm{Re}\langle \phi(0)\phi(t) \rangle$";markers_dict(5,:dtriangle)...)
    scatter!(fig,tp,corr0tIm .± err_corr0tIm,label=L"$\textrm{Im}\langle \phi(0)\phi(t) \rangle$";markers_dict(6,:utriangle)...)
    
    return fig
end