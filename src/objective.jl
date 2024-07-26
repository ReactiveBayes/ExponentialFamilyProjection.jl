
"""
    ProjectionCostGradientObjective

This structure provides an interface for `Manopt` to compute the cost and gradients required for the optimization procedure based on manifold projection. The actual computation of costs and gradients is defined by the `strategy` argument.

# Arguments

- `projection_parameters`: The parameters for projection, must be of type `ProjectionParameters`
- `projection_argument`: The second argument of the `project_to` function.
- `supplementary_η`: A tuple of additional natural parameters subtracted from the current estimated value in each optimization iteration.
- `strategy`: Specifies the method for computing costs and gradients, which may support different `projection_argument` values.
- `strategy_state`: The state for the `strategy`, usually created with `create_state!`
- `buffer`: Optional; some strategies may use this to optimize memory allocation.

!!! note
    This structure is internal and subject to change.
"""
struct ProjectionCostGradientObjective{J,F,P,S,T,B}
    projection_parameters::J
    projection_argument::F
    supplementary_η::P
    strategy::S
    strategy_state::T
    buffer::B
end

get_projection_parameters(obj::ProjectionCostGradientObjective) = obj.projection_parameters
get_projection_argument(obj::ProjectionCostGradientObjective) = obj.projection_argument
get_supplementary_η(obj::ProjectionCostGradientObjective) = obj.supplementary_η
get_strategy(obj::ProjectionCostGradientObjective) = obj.strategy
get_strategy_state(obj::ProjectionCostGradientObjective) = obj.strategy_state
get_buffer(obj::ProjectionCostGradientObjective) = obj.buffer

function (objective::ProjectionCostGradientObjective)(M::AbstractManifold, X, p)
    ef = convert(ExponentialFamilyDistribution, M, p)

    strategy = get_strategy(objective)
    state = get_strategy_state(objective)
    projection_parameters = get_projection_parameters(objective)
    projection_argument = get_projection_argument(objective)
    supplementary_η = get_supplementary_η(objective)

    state = prepare_state!(
        strategy,
        state,
        M,
        projection_parameters,
        projection_argument,
        ef,
        supplementary_η,
    )

    logpartition = ExponentialFamily.logpartition(ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(ef)
    inv_fisher = cholinv(ExponentialFamily.fisherinformation(ef))
    η = copy(ExponentialFamily.getnaturalparameters(ef))

    # If we have some supplementary natural parameters in the objective 
    # we must subtract them from the natural parameters of the current η
    foreach(supplementary_η) do s_η
        vmap!(-, η, η, s_η)
    end

    c = compute_cost(
        strategy,
        state,
        objective,
        M,
        η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )
    X = compute_gradient!(
        strategy,
        state,
        objective,
        M,
        X,
        η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )
    X = project!(M, X, p, X)

    return c, X
end


