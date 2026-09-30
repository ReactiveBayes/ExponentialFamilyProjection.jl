# Exact categorical expectations still use the K-1 independent natural coordinates.
ExponentialFamilyProjection.prepare_inv_fisher(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    ::ClosedFormStrategy,
    current_ef,
) where {F} = nothing

function ExponentialFamilyProjection.preprocess_strategy_argument(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    strategy::ClosedFormStrategy,
    argument,
) where {F}
    if argument isa AbstractArray
        throw(
            ArgumentError(
                "`ClosedFormStrategy` for `Categorical` requires a callable log-score or a distribution, not samples.",
            ),
        )
    elseif argument isa Union{Distribution,ProductOf}
        return (strategy, Logpdf(argument))
    end
    # Unlike other families, finite categorical expectations support arbitrary
    # scalar callables. Do not reinterpret a closure's captures as distributions.
    return (strategy, argument)
end

struct CategoricalClosedFormStrategyState{T,Q}
    target::T
    current_ef::Q
end

function ExponentialFamilyProjection.create_state!(
    strategy::ClosedFormStrategy,
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    parameters::ProjectionParameters,
    projection_argument,
    initial_ef,
    supplementary_η,
) where {F}
    # Own a flat parameter buffer, independent of whether the caller uses a
    # vector or the manifold's partitioned representation.
    current_ef = ExponentialFamilyDistribution(
        Categorical,
        collect(getnaturalparameters(initial_ef)),
        getconditioner(initial_ef),
    )
    return CategoricalClosedFormStrategyState(projection_argument, current_ef)
end

function ExponentialFamilyProjection.prepare_state!(
    ::ClosedFormStrategy,
    state::CategoricalClosedFormStrategyState,
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    parameters::ProjectionParameters,
    projection_argument,
    current_ef,
    supplementary_η,
) where {F}
    # Keep the actual current distribution: the objective subsequently subtracts
    # supplementary parameters from η, but expectations must still be under q.
    copyto!(getnaturalparameters(state.current_ef), getnaturalparameters(current_ef))
    return state
end

function ExponentialFamilyProjection.compute_cost(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    ::ClosedFormStrategy,
    state::CategoricalClosedFormStrategyState,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
) where {F}
    expected_target = mean(ClosedFormExpectation(), state.target, state.current_ef)
    # Categorical base measures are one. Fixed supplementary normalization
    # constants are omitted, as in ControlVariateStrategy's objective.
    return dot(gradlogpartition, η) - logpartition - expected_target
end

function ExponentialFamilyProjection.compute_gradient!(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    strategy::ClosedFormStrategy,
    state::CategoricalClosedFormStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
) where {F}
    gradient = mean(ClosedWilliamsProduct(strategy.backend), state.target, state.current_ef)
    n = length(gradlogpartition) - 1
    natural_gradient = similar(gradient, n)
    ExponentialFamilyProjection.categorical_inv_fisher_mul!(
        natural_gradient,
        gradlogpartition,
        view(gradient, 1:n),
    )
    for i = 1:n
        X[i] = η[i] - natural_gradient[i]
    end
    X[end] = zero(eltype(X))
    return X
end
