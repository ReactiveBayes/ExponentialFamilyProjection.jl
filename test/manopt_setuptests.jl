using Manopt

# Non allocating version of the same `StopWhenGradientNormLess` from `Manopt.jl`
struct StopWhenGradientNormLessNonAllocating{F} <: StoppingCriterion
    threshold::F
end

function (sc::StopWhenGradientNormLessNonAllocating)(mp, s, i)
    M = get_manifold(mp)
    if (i > 0)
        grad_norm = norm(M, get_iterate(s), get_gradient(s))
        if grad_norm < sc.threshold
            return true
        end
    end
    return false
end

# Non allocating version of the same `ConstantStepsize` from `Manopt.jl`
struct ConstantStepsizeNonAllocating{T} <: Stepsize
    stepsize::T
end
function (cs::ConstantStepsizeNonAllocating)(args...; kwargs...)
    return cs.stepsize
end