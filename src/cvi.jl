
struct CVICostGradientObjective{F,S,B}
    targetfn::F
    strategy::S
    buffer::B
end

get_cvi_targetfn(obj::CVICostGradientObjective) = obj.targetfn
get_cvi_strategy(obj::CVICostGradientObjective) = obj.strategy
get_cvi_buffer(obj::CVICostGradientObjective) = obj.buffer

function (obj::CVICostGradientObjective)(M::AbstractManifold, X, p)
    ef = convert(ExponentialFamilyDistribution, M, p)

    strategy = get_cvi_strategy(obj)
    state = prepare_state!(strategy, obj.targetfn, ef)

    logpartition = ExponentialFamily.logpartition(ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(ef)
    η = ExponentialFamily.getnaturalparameters(ef)
    inv_fisher = cholinv(ExponentialFamily.fisherinformation(ef))

    c = compute_cost(obj, strategy, state, η, logpartition, gradlogpartition, inv_fisher)
    X = compute_gradient!(obj, strategy, state, X, η, logpartition, gradlogpartition, inv_fisher)
    X = project!(M, X, p, X)

    return c, X
end


