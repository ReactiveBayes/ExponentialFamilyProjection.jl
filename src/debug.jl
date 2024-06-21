import Manopt: AbstractManoptProblem, AbstractManoptSolverState, ManifoldCostGradientObjective
import Manopt: DebugAction
export DebugCovGradient
using Printf

@doc raw"""
    DebugCovGradient <: DebugAction

print the current cov of the gradient value, see [`get_cost`](@ref).

# Constructors
    DebugCovGradient()

# Parameters

* `format`: (`"$prefix %f"`) format to print the output
* `io`:     (`stdout`) default stream to print the debug to.
* `long`:   (`false`) short form to set the format to `f(x):` (default) or `current cost: ` and the cost
"""
mutable struct DebugCovGradient <: DebugAction
    io::IO
    format::String
    function DebugCovGradient(;
        long::Bool=false, io::IO=stdout, format=long ? "current gradient cov: [" : "Cov(grad): ["
    )
        return new(io, format)
    end
end
function (d::DebugCovGradient)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int)
    if i >= 0
        obj = get_objective(p)
        cov_matrix = get_cov_gradient(obj, get_iterate(st))
        Printf.format(d.io, Printf.Format(d.format))
        for r in eachrow(cov_matrix)
            foreach(x -> @printf(d.io, " %f ", x), r)
            @printf(d.io, ";")
        end
        @printf(d.io, "]")
    end
    return nothing
end
function show(io::IO, di::DebugCovGradient)
    return print(io, "DebugCovGradient(; format=\"$(escape_string(di.format))\")")
end
status_summary(di::DebugCovGradient) = "(:Cov(grad), \"$(escape_string(di.format))\")"

function get_cov_gradient(obj::ManifoldCostGradientObjective, p)
    cvi = obj.costgrad!!
    strategy = get_cvi_strategy(cvi)
    state = getstate(strategy)
    cov_matrix = cov(state.gradsamples', state.gradsamples')
    return cov_matrix
end
