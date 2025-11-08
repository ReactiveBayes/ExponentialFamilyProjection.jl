
"""
    MLEStrategy()

A strategy for gradient descent optimization and gradients computations that resembles MLE estimation.

!!! note
    This strategy requires a collection of samples as an argument for `project_to` and cannot project a function. Use `ControlVariateStrategy` to project a function.
"""
struct MLEStrategy end

preprocess_strategy_argument(strategy::MLEStrategy, argument::AbstractArray) =
    (strategy, argument)
preprocess_strategy_argument(::MLEStrategy, argument::Any) = error(
    lazy"`MLEStrategy` requires the projection argument to be an array of samples. Got `$(typeof(argument))` instead.",
)

Base.@kwdef struct MLEStrategyState{F}
    targetfn::F
end

function Base.:(==)(a::MLEStrategyState, b::MLEStrategyState)::Bool
    return a.targetfn == b.targetfn
end

gettargetfn(state::MLEStrategyState) = state.targetfn

function create_state!(
    strategy::MLEStrategy,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    samples::AbstractArray,
    initial_ef,
    supplementary_η,
)
    _, sample_container = ExponentialFamily.check_logpdf(initial_ef, samples)

    sufficientstatistics = zeros(
        paramfloattype(initial_ef),
        length(getnaturalparameters(initial_ef)),
        length(sample_container),
    )

    J = size(sufficientstatistics, 1)

    foreach(enumerate(sample_container)) do (i, sample)
        sample_sufficientstatistics = __projection_fast_pack_parameters(
            ExponentialFamily.sufficientstatistics(initial_ef, sample),
        )
        for j = 1:J
            @inbounds sufficientstatistics[j, i] = sample_sufficientstatistics[j]
        end
    end

    targetfn = MLETargetFn(M, samples, sufficientstatistics)
    return MLEStrategyState(targetfn)
end

function prepare_state!(
    strategy::MLEStrategy,
    state::MLEStrategyState,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    samples::AbstractArray,
    current_ef,
    supplementary_η,
)
    return state
end

struct MLETargetFn{M,S,C}
    manifold::M
    samples::S
    sufficientstatistics::C
end

function (fn::MLETargetFn)(η)
    # This function essentially computes the negative average of `logpdf` of all provided `samples`
    # with the distribution defined in `η`
    ef = convert(
        ExponentialFamilyDistribution,
        fn.manifold,
        ExponentialFamilyManifolds.partition_point(fn.manifold, η),
    )
    _, samples_container = ExponentialFamily.check_logpdf(ef, fn.samples)
    # We use precomputed `sufficientstatistics` since in this strategy the `samples` are fixed
    sufficientstatistics_container = eachcol(fn.sufficientstatistics)
    _logpartition = logpartition(ef)
    return -mean(
        zip(samples_container, sufficientstatistics_container),
    ) do (sample, sufficientstatistics)
        _logbasemeasure = logbasemeasure(ef, sample)
        # This is the definition of `logpdf` for exponential family members
        return _logbasemeasure + dot(η, sufficientstatistics) - _logpartition
    end
end

function compute_cost(
    M::AbstractManifold,
    strategy::MLEStrategy,
    state::MLEStrategyState,
    η,
    _,
    _,
    _,
)
    return gettargetfn(state)(η)
end

function compute_gradient!(
    M::AbstractManifold,
    strategy::MLEStrategy,
    state::MLEStrategyState,
    X,
    η,
    _,
    _,
    inv_fisher,
)
    ef = convert(
        ExponentialFamilyDistribution,
        M,
        ExponentialFamilyManifolds.partition_point(M, η),
    )
    targetfn = gettargetfn(state)
    mean_sufficient_stats = mean(targetfn.sufficientstatistics, dims = 2)
    G = -view(mean_sufficient_stats, :, 1) + gradlogpartition(ef)
    X = mul!(X, inv_fisher, G)
    return X
end
