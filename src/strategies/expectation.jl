export ExpectationStrategy

"""
    ExpectationStrategy(backend)

A deterministic strategy: the expectations of the projection are a quadrature from `backend`, a
backend of BayesBase's expectation protocol, such as ExpectationApproximations' `EADefault()`.
At every iteration the backend gives a point set for the current member of the family, weighted
points `xᵢ, wᵢ`, and

- the gradient is `Σᵢ wᵢ (log p̃(xᵢ) - log h(xᵢ)) (T(xᵢ) - ∇A(η))`, the covariance of the
  sufficient statistics with the target's log-density, whose natural gradient, through the
  inverse of the Fisher information, gives the step;
- the cost is the negative evidence lower bound, `-Σᵢ wᵢ log p̃(xᵢ) - H[q]`.

It needs no random numbers: the same target gives the same projection, and the gradient has no
variance, so the iterations can stop at a tight tolerance. The points depend only on the current
member, never on the target, so the quadrature must suit the family: Gauss–Hermite for Gaussians,
Gauss–Laguerre for Gamma, Gauss–Jacobi for Beta, as ExpectationApproximations chooses.

The argument of `project_to` must be callable, the unnormalised log-density.
"""
struct ExpectationStrategy{B}
    backend::B
end

Base.:(==)(a::ExpectationStrategy, b::ExpectationStrategy) = a.backend == b.backend

preprocess_strategy_argument(strategy::ExpectationStrategy, argument::Any) = (strategy, argument)
preprocess_strategy_argument(::ExpectationStrategy, argument::AbstractArray) = error(
    lazy"The `ExpectationStrategy` requires the projection argument to be a callable object (e.g. `Function`). Got `$(typeof(argument))` instead.",
)

# The quadrature of the current member: its weights, and for each point the target's and the base
# measure's log-densities and the sufficient statistics
mutable struct ExpectationStrategyState{W, L, S}
    weights::W
    logpdfs::L
    logbasemeasures::L
    sufficientstatistics::S
end

function create_state!(strategy::ExpectationStrategy, M::AbstractManifold, parameters::ProjectionParameters, projection_argument, initial_ef, supplementary_η)
    state = ExpectationStrategyState(Float64[], Float64[], Float64[], zeros(0, 0))
    return prepare_state!(strategy, state, M, parameters, projection_argument, initial_ef, supplementary_η)
end

function prepare_state!(strategy::ExpectationStrategy, state::ExpectationStrategyState, M::AbstractManifold, parameters::ProjectionParameters, projection_argument, current_ef, supplementary_η)
    distribution = convert(Distribution, current_ef)
    chosen = BayesBase.which_expectation_backend(strategy.backend, Any, typeof(distribution))
    isnothing(chosen) && throw(ArgumentError(lazy"`ExpectationStrategy`'s backend has no quadrature for a `$(typeof(distribution))`"))
    prepared = BayesBase.prepare_expectation(chosen, distribution)
    prepared isa BayesBase.ExpectationPointSet ||
        throw(ArgumentError(lazy"`ExpectationStrategy` needs a point set from its backend, got a `$(typeof(prepared))`"))
    points, weights = prepared.points, prepared.weights
    n = length(points)
    J = length(getnaturalparameters(current_ef))
    resize!(state.weights, n)
    resize!(state.logpdfs, n)
    resize!(state.logbasemeasures, n)
    size(state.sufficientstatistics) == (J, n) || (state.sufficientstatistics = zeros(J, n))
    # the supplementary members are the same family: each brings its base measure once more
    one_minus_n_of_supplementary = 1 - length(supplementary_η)
    for (i, x) in enumerate(points)
        @inbounds state.weights[i] = weights[i]
        @inbounds state.logpdfs[i] = projection_argument(x)
        @inbounds state.logbasemeasures[i] = one_minus_n_of_supplementary * ExponentialFamily.logbasemeasure(current_ef, x)
        statistics = __projection_fast_pack_parameters(ExponentialFamily.sufficientstatistics(current_ef, x))
        for j in 1:J
            @inbounds state.sufficientstatistics[j, i] = statistics[j]
        end
    end
    return state
end

function compute_cost(M::AbstractManifold, strategy::ExpectationStrategy, state::ExpectationStrategyState, η, logpartition, gradlogpartition, inv_fisher)
    return dot(gradlogpartition, η) - dot(state.weights, state.logpdfs) - logpartition + dot(state.weights, state.logbasemeasures)
end

function compute_gradient!(M::AbstractManifold, strategy::ExpectationStrategy, state::ExpectationStrategyState, X, η, logpartition, gradlogpartition, inv_fisher)
    J = length(gradlogpartition)
    gradient = zeros(J)
    for i in eachindex(state.weights)
        @inbounds residual = state.weights[i] * (state.logpdfs[i] - state.logbasemeasures[i])
        for j in 1:J
            @inbounds gradient[j] += residual * (state.sufficientstatistics[j, i] - gradlogpartition[j])
        end
    end
    X .= η .- inv_fisher * gradient
    return X
end
