using JLD2, DataFrames, Distributions, Optim, Plots, StatsFuns, BenchmarkTools, SPGBox, ForwardDiff, PGFPlotsX, Colors

# Import Data -> DataFrame
df = load("data/deg_d10_0006.jld2", "data")

function IR(ψ)
    return select(filter(row -> row."Step_Index" == ψ[1] || row."Step_Index" == ψ[2] , df), "Cycle_Index", "Internal_Resistance(Ohm)")
end

y = IR([11,28]) # Filter DataFrame for Resistance Measurements
 

# Polynomial Generation
function gen(n, A, β, X)
    ϵ = randn(n) * A
    return (β[1] .+ X * β[2] .+ X.^2 * β[3]) .+ ϵ
end

# Log Likelihood
function log_lkhd(X, y, n, β, A) 
    ŷ = gen(n,A,β,X)
    return -sum(logpdf.(Normal(0,1),y-ŷ))
end

# Find Coefficents
function Coeffs(nvar, n, A, X, y, β̂)
    # res = optimize(β -> log_lkhd(X, y, n, β, A), [0.,0.,0.], LBFGS(), autodiff=:forward, Optim.Options(allow_f_increases=true))
    # return Optim.minimizer(res)
    spgbox!(β -> log_lkhd(X, y, n, β, A),(g,β) -> ForwardDiff.gradient!(g,β -> log_lkhd(X, y, n, β, A),β), β̂)#, nitmax = 200)
end


# Main Call
nvar,n = 2,201
A = 0. # Noise Amplitude
β̂ = zeros(3)

# X = rand(Normal(0,1), n, nvar-1)
#y = gen(n,A,β,X)
# β̂ = Coeffs(nvar,n,A,β,X,y)
Coeffs(nvar,n,A,y[:,1],y[:,2],β̂)

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
ŷ = gen(n,0.,β̂,y[:,1])
ψ = sortslices([y[:,1][:] ŷ[:]], dims=1)

κ = distinguishable_colors(10)
@pgf Axis(
    {
        xlabel = "Cycle Number",
        ylabel = "DC Internal Resistance (Ω)",
        clip_mode="individual",
        title = "Maximum Likelihood Coefficents for DCIR (Sony VTC6)",
        height = "10.5cm", 
        width = "14cm",
        xmin = 0,
        xmax = 200
    },
    Plot({"only marks", color=κ[10]},
        Table([:x => y[:,1], :y =>y[:,2]])
        ),
    Plot({color=κ[5], "thick"},
    Table([:x => ψ[:,1], :y =>ψ[:,2]])
    )

)