using  Distributions, Optim, Plots, StatsFuns, BenchmarkTools, SPGBox, ForwardDiff, PGFPlotsX, Colors

# Data Generation
function gen(n, A, β, X)
    ϵ = randn(n) * A
    return (β[1] .+ X * β[2] .+ X.^2 * β[3] .+ X.^3 * β[4]) .+ ϵ
end

# Log Likelihood
function log_lkhd(X, y, n, β, A) 
    ŷ = gen(n,A,β,X)
    return -sum(logpdf.(Normal(0,1),y-ŷ))/n
end


# Find Coefficents
function Coeffs(nvar, n, A, X, y,β̂)
    # res = optimize(β -> log_lkhd(X, y, n, β, A), [0.,0.,0.], LBFGS(), autodiff=:forward, Optim.Options(allow_f_increases=true))
    # return Optim.minimizer(res)
    spgbox!(β -> log_lkhd(X, y, n, β, A),(g,β) -> ForwardDiff.gradient!(g,β -> log_lkhd(X, y, n, β, A),β), β̂, nitmax = 100)
end


# Main Call
nvar,n = 2,201
A = 1. # Noise Amplitude
β = [0.5,1.6,7,3.2] # Polynomial coeffs.
β̂ = zeros(4)

X = rand(Normal(0,1), n, nvar-1)
y = gen(n,A,β,X)
Coeffs(nvar,n,A,X,y,β̂)

# @benchmark Coeffs(nvar,n,A,β,X,y,β̂)
# bₜ = @benchmarkable Coeffs(a,b,c,d,e,f,g) setup = begin;
#     a = copy($nvar);
#     b = copy($n);
#     c = copy($A);
#     d = copy($β);
#     e = copy($X);
#     f = copy($y);
#     g = copy($β̂);
# end
# run(bₜ, evals=100, seconds=10, samples=1000) 


# Assess Results
# μ =  β-β̂
ŷ = gen(n,0.,β̂,X)
ψ = sortslices([X[:] ŷ[:]], dims=1)

κ = distinguishable_colors(10)
@pgf Axis(
    {
        xlabel = "Input",
        ylabel = "Ouput",
        clip_mode="individual",
        title = "Maximum Likelihood Coefficents",
        height = "10.5cm", 
        width = "14cm",
    },
    Plot({"only marks", color=κ[10]},
        Table([:x => X, :y =>y])
        ),
    Plot({color=κ[5], "thick"},
    Table([:x => ψ[:,1], :y =>ψ[:,2]])
    )
)