
struct CVICostGradientObjective{F,P,S,B}
    targetfn::F
    supplementary_η::P
    strategy::S
    buffer::B
end

get_cvi_targetfn(obj::CVICostGradientObjective) = obj.targetfn
get_cvi_supplementary_η(obj::CVICostGradientObjective) = obj.supplementary_η
get_cvi_strategy(obj::CVICostGradientObjective) = obj.strategy
get_cvi_buffer(obj::CVICostGradientObjective) = obj.buffer

function (objective::CVICostGradientObjective)(M::AbstractManifold, X, p)
    ef = convert(ExponentialFamilyDistribution, M, p)

    strategy = get_cvi_strategy(objective)
    state = prepare_state!(strategy, objective.targetfn, ef, objective.supplementary_η)

    logpartition = ExponentialFamily.logpartition(ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(ef)
    inv_fisher = cholinv(ExponentialFamily.fisherinformation(ef))
    η = copy(ExponentialFamily.getnaturalparameters(ef))

    # If we have some supplementary natural parameters in the objective 
    # we must subtract them from the natural parameters of the current η
    supplementary = get_cvi_supplementary_η(objective)
    foreach(supplementary) do s_η
        vmap!(-, η, η, s_η)
    end

    c = compute_cost(
        objective,
        strategy,
        state,
        η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )
    X = compute_gradient!(
        objective,
        strategy,
        state,
        X,
        η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )
    X = project!(M, X, p, X)

    return c, X
end


