import Manopt

using LoopVectorization

"""
    BoundedNormUpdateRule(limit; direction = IdentityUpdateRule()) 

A `DirectionUpdateRule` that bounds the norm of the gradient to a specific value (`limit`).
First calls the `direction` update rule and then modifies (if necessary) the result in place.
"""
struct BoundedNormUpdateRule{L,D} <: Manopt.DirectionUpdateRule
    limit::L
    direction::D
end

function BoundedNormUpdateRule(limit; direction = IdentityUpdateRule())
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

