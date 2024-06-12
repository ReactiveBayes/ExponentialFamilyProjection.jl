
struct CVICostGradientObjective{F,S}
    targetfn::F
    strategy::S
end

get_cvi_targetfn(obj::CVICostGradientObjective) = obj.targetfn
get_cvi_strategy(obj::CVICostGradientObjective) = obj.strategy

function (obj::CVICostGradientObjective)(M::AbstractManifold, X, p)
    ef = convert(ExponentialFamilyDistribution, M, p)

    strategy = get_cvi_strategy(obj)
    state = prepare_state!(strategy, obj.targetfn, ef)

    logpartition = ExponentialFamily.logpartition(ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(ef)
    η = ExponentialFamily.getnaturalparameters(ef)
    inv_fisher = cholinv(ExponentialFamily.fisherinformation(ef))

    c = compute_cost(strategy, state, η, logpartition, gradlogpartition, inv_fisher)
    X = compute_gradient!(strategy, state, X, η, logpartition, gradlogpartition, inv_fisher)
    X = project!(M, X, p, X)

    return c, X
end

