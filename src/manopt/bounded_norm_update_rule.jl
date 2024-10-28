import Manopt

using LoopVectorization

"""
    BoundedNormUpdateRule(limit; direction = Manopt.IdentityUpdateRule()) 

A `DirectionUpdateRule` is a direction rule that constrains the norm of the direction to a specified limit.

This rule operates in two steps:
    
- Initial direction computation: It first applies the specified `direction` update rule to compute an initial direction.
- Norm check and scaling: The norm of the resulting direction vector is checked using `Manopt.norm(M, p, d)`, where:
  - `M`` is the manifold on which the optimization is running,
  - `p` is the point at which the direction was computed,
  - `d` is the computed direction.
  - If this norm exceeds the specified `limit`, the direction vector is scaled down so that its new norm exactly equals the limit. This scaling preserves the direction of the gradient while controlling its magnitude.

Read more about `Manopt.DirectionUpdateRule` in the `Manopt.jl` documentation.
"""
struct BoundedNormUpdateRule{L,D} <: Manopt.DirectionUpdateRule
    limit::L
    direction::D
end

function BoundedNormUpdateRule(limit; direction = Manopt.IdentityUpdateRule())
    return BoundedNormUpdateRule(limit, direction)
end

function (b::BoundedNormUpdateRule)(
    mp::Manopt.AbstractManoptProblem,
    s::Manopt.AbstractGradientSolverState,
    i,
)
    M = Manopt.get_manifold(mp)
    p = Manopt.get_iterate(s)
    step, d = b.direction(mp, s, i)
    C = Manopt.norm(M, p, d)
    if C > b.limit
        @turbo warn_check_args = false for i in eachindex(d)
            d[i] = d[i] * b.limit / C
        end
    end
    return step, d
end

