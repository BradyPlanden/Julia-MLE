using JLD2, DataFrames, Distributions, Plots, BenchmarkTools, SPGBox, ForwardDiff, PGFPlotsX, Colors
"""
This script computes the maximum likelihood estimation for polynomial coefficients of second-order 
given an experimental battery dataset capturing DCIR for multiple cycles.  

"""

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
function log_lkhd(n, A, β, X, y) 
    ŷ = gen(n,A,β,X)
    return -sum(logpdf.(Normal(0,1),y-ŷ))
end

# Maximum Likelihood Estimation (Mutated in-place)
function Coeffs!(β̂, n, A, X, y, nvar)
    spgbox!(β -> log_lkhd(n, A, β, X, y),(g,β) -> ForwardDiff.gradient!(g,β -> log_lkhd(n, A, β, X, y),β), β̂, nitmax = 100)
end


# Initialise variables
nvar,n = 2,201
A = 0. # Noise Amplitude
β̂ = zeros(3)

# MLE + Benchmark
@show @benchmark Coeffs!(β̂, n, A, y[:,1], y[:,2], nvar)

# Assess Results
ŷ = gen(n,0.,β̂,y[:,1])
ψ = sortslices([y[:,1][:] ŷ[:]], dims=1)

# Plot using PGFPlots for Presentation
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