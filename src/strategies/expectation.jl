export ExpectationStrategy

"""
    ExpectationStrategy(backend)

A deterministic projection strategy that computes the expectations of the projection with
`backend`, a backend of BayesBase's expectation protocol, such as ExpectationApproximations'
`EADefault()`. At every iteration the backend returns a point set for the current member of the
family, the points `xᵢ` with weights `wᵢ`. From these:

- the step is towards the natural parameters `Cov_w(T, T)⁻¹ Cov_w(T, g)`, the weighted regression
  of `g = log p̃ - log h` on the sufficient statistics `T`, with both covariances taken over the
  same points. This is the natural-gradient step, with the Fisher information replaced by its
  quadrature. It does not change when a constant is added to `log p̃`, and a target in the family
  is projected onto itself, whatever the accuracy of the quadrature on `T`;
- the cost is the negative evidence lower bound, `-Σᵢ wᵢ log p̃(xᵢ) - H[q]`.

The strategy draws no samples, so the step has no variance. Its accuracy is that of the
quadrature on the target's log-density. The points depend on the current member of the family
and not on the target, so the backend's quadrature must suit the family. ExpectationApproximations,
for example, uses Gauss–Hermite quadrature for Gaussians, capped in the number of points as the
dimension grows, and Gauss–Legendre quadrature in `log x` for Gamma and in `logit x` for Beta
distributions, which also holds at shapes below 1, where the density is singular at the boundary.

The argument of `project_to` must be callable: it is the unnormalised log-density of the target.
A distribution is not callable; pass `Base.Fix1(logpdf, distribution)`.
"""
struct ExpectationStrategy{B}
    backend::B
end

Base.:(==)(a::ExpectationStrategy, b::ExpectationStrategy) = a.backend == b.backend

preprocess_strategy_argument(strategy::ExpectationStrategy, argument::Any) =
    (strategy, argument)
preprocess_strategy_argument(::ExpectationStrategy, argument::AbstractArray) = error(
    lazy"The `ExpectationStrategy` requires the projection argument to be a callable object (e.g. `Function`). Got `$(typeof(argument))` instead.",
)
preprocess_strategy_argument(::ExpectationStrategy, argument::Distribution) = error(
    lazy"The `ExpectationStrategy` requires the projection argument to be a callable log-density. Got a `$(typeof(argument))`; pass `Base.Fix1(logpdf, distribution)` instead.",
)

# The quadrature of the current member: its weights, and for each point the target's and the base
# measure's log-densities and the sufficient statistics
mutable struct ExpectationStrategyState{W,L,S}
    weights::W
    logpdfs::L
    logbasemeasures::L
    sufficientstatistics::S
end

function create_state!(
    strategy::ExpectationStrategy,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    initial_ef,
    supplementary_η,
)
    state = ExpectationStrategyState(Float64[], Float64[], Float64[], zeros(0, 0))
    return prepare_state!(
        strategy,
        state,
        M,
        parameters,
        projection_argument,
        initial_ef,
        supplementary_η,
    )
end

function prepare_state!(
    strategy::ExpectationStrategy,
    state::ExpectationStrategyState,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    current_ef,
    supplementary_η,
)
    distribution = convert(Distribution, current_ef)
    chosen =
        BayesBase.which_expectation_backend(strategy.backend, Any, typeof(distribution))
    isnothing(chosen) && throw(
        ArgumentError(
            lazy"`ExpectationStrategy`'s backend has no quadrature for a `$(typeof(distribution))`",
        ),
    )
    prepared = BayesBase.prepare_expectation(chosen, distribution)
    prepared isa BayesBase.ExpectationPointSet || throw(
        ArgumentError(
            lazy"`ExpectationStrategy` needs a point set from its backend, got a `$(typeof(prepared))`",
        ),
    )
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
        @inbounds state.logbasemeasures[i] =
            one_minus_n_of_supplementary * ExponentialFamily.logbasemeasure(current_ef, x)
        statistics = __projection_fast_pack_parameters(
            ExponentialFamily.sufficientstatistics(current_ef, x),
        )
        for j = 1:J
            @inbounds state.sufficientstatistics[j, i] = statistics[j]
        end
    end
    return state
end

function compute_cost(
    M::AbstractManifold,
    strategy::ExpectationStrategy,
    state::ExpectationStrategyState,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    return dot(gradlogpartition, η) - dot(state.weights, state.logpdfs) - logpartition +
           dot(state.weights, state.logbasemeasures)
end

# The step uses the quadrature's covariance of the sufficient statistics, not the analytic Fisher
# information, so that the two covariances of the regression are taken over the same points
prepare_inv_fisher(::AbstractManifold, ::ExpectationStrategy, current_ef) = nothing

# `η - Cov_w(T, T)⁻¹ Cov_w(T, g)`, with `g = log p̃ - log h`: centring `g` removes the constant of
# the target, and taking both covariances over the points makes a target in the family exact
function compute_gradient!(
    M::AbstractManifold,
    strategy::ExpectationStrategy,
    state::ExpectationStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    weights, S = state.weights, state.sufficientstatistics
    J, n = size(S)
    ḡ = zero(eltype(weights))
    T̄ = zeros(J)
    for i = 1:n
        @inbounds ḡ += weights[i] * (state.logpdfs[i] - state.logbasemeasures[i])
        for j = 1:J
            @inbounds T̄[j] += weights[i] * S[j, i]
        end
    end
    CTT = zeros(J, J)
    CTg = zeros(J)
    for i = 1:n
        @inbounds gᵢ = state.logpdfs[i] - state.logbasemeasures[i] - ḡ
        for j = 1:J
            @inbounds tj = S[j, i] - T̄[j]
            @inbounds CTg[j] += weights[i] * tj * gᵢ
            for k = 1:J
                @inbounds CTT[k, j] += weights[i] * (S[k, i] - T̄[k]) * tj
            end
        end
    end
    X .= η .- (Symmetric(CTT) \ CTg)
    return X
end
