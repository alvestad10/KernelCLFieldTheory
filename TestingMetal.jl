using Metal
using BenchmarkTools
using LinearAlgebra

Metal.versioninfo()

N = 100
A = randn(Float32,N,N);
B = randn(Float32,N,N);
C = zeros(Float32,N,N);

@btime mul!($C,$A,$B);
@btime @. C = 2*$A*$B;

_A = MtlArray(A);
_B = MtlArray(B);
_C = MtlArray(C);

@btime _C = $_A*$_B;
#@btime mul!($_C,$_A,$_B);
@btime @. _C = 2* $_A * $_B;


using StochasticDiffEq


f_mtl(du,u,p,t) = (du .= _A*u)
g_mtl(du,u,p,t) = (du .= _B)

u0_mtl = Metal.ones(N)*0.001f0
prob_mtl = SDEProblem(f_mtl,g_mtl,u0_mtl,(0.0f0,1.0f0),
                                noise_rate_prototype=_C
            )

@btime solve(prob_mtl,LambaEM());


f_cpu(du,u,p,t) = mul!(du,A,u)
g_cpu(du,u,p,t) = (du .= B)

u0_cpu = ones(Float32,N)*0.001f0
prob_cpu = SDEProblem(f_cpu,g_cpu,u0_cpu,(0.0f0,1.0f0),
                                noise_rate_prototype=C
            )

@btime solve($prob_cpu,LambaEM());