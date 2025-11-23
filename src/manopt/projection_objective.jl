"""
    ProjectionCostGradientObjective

This structure provides an interface for `Manopt` to compute the cost and gradients required for the optimization procedure based on manifold projection. The actual computation of costs and gradients is defined by the `strategy` argument.

# Arguments

- `projection_parameters`: The parameters for projection, must be of type `ProjectionParameters`
- `projection_argument`: The second argument of the `project_to` function.
- `current_η`: Current optimization point.
- `supplementary_η`: A tuple of additional natural parameters subtracted from the current point in each optimization iteration.
- `strategy`: Specifies the method for computing costs and gradients, which may support different `projection_argument` values.
- `strategy_state`: The state for the `strategy`, usually created with `create_state!`

!!! note
    This structure is internal and is subject to change.
"""
struct ProjectionCostGradientObjective{J,F,C,P,S,T}
    projection_parameters::J
    projection_argument::F
    current_η::C
    supplementary_η::P
    strategy::S
    strategy_state::T
end

get_projection_parameters(obj::ProjectionCostGradientObjective) = obj.projection_parameters
get_projection_argument(obj::ProjectionCostGradientObjective) = obj.projection_argument
get_current_η(obj::ProjectionCostGradientObjective) = obj.current_η
get_supplementary_η(obj::ProjectionCostGradientObjective) = obj.supplementary_η
get_strategy(obj::ProjectionCostGradientObjective) = obj.strategy
get_strategy_state(obj::ProjectionCostGradientObjective) = obj.strategy_state

function call_objective(
    objective::ProjectionCostGradientObjective,
    M::AbstractManifold,
    X,
    p,
)
    current_ef = convert(ExponentialFamilyDistribution, M, p)
    current_η = copyto!(get_current_η(objective), getnaturalparameters(current_ef))

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
        current_ef,
        supplementary_η,
    )

    logpartition = ExponentialFamily.logpartition(current_ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(current_ef)
    inv_fisher = cholinv(ExponentialFamily.fisherinformation(current_ef))

    # If we have some supplementary natural parameters in the objective 
    # we must subtract them from the natural parameters of the current η
    foreach(supplementary_η) do s_η
        map!(-, current_η, current_η, s_η)
    end

    c = compute_cost(
        M,
        strategy,
        state,
        current_η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )

    X_nat = compute_gradient!(
        M,
        strategy,
        state,
        X,
        current_η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )
    X = jacobian_nat_to_manifold!(M, X, X_nat)
    X = project!(M, X, p, X)
    return c, X
end

function (objective::ProjectionCostGradientObjective)(M::AbstractManifold, X, p)
    return call_objective(objective, M, X, p)
end