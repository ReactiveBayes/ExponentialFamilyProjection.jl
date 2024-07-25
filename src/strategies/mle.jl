using ForwardDiff, LoopVectorization

Base.@kwdef struct MLEStrategy{D,N,T}
    seed::D = 42
    rng::N = StableRNG(seed)
    state::T = nothing
end

getseed(strategy::MLEStrategy) = strategy.seed
getrng(strategy::MLEStrategy) = strategy.rng
getstate(strategy::MLEStrategy) = strategy.state

function Base.:(==)(a::MLEStrategy, b::MLEStrategy)::Bool
    return getseed(a) == getseed(b) && getrng(a) == getrng(b) && getstate(a) == getstate(b)
end

function getinitialpoint(strategy::MLEStrategy, M::AbstractManifold)
    return rand(getrng(strategy), M)
end

function with_state(strategy::MLEStrategy, state)
    return MLEStrategy(seed = getseed(strategy), rng = getrng(strategy), state = state)
end

preprocess_strategy_argument(strategy::MLEStrategy, argument::AbstractArray) = strategy
preprocess_strategy_argument(::MLEStrategy, argument::Any) = error(
    lazy"`MLEStrategy` requires the projection argument to be an array of samples. Got `$(typeof(argument))` instead.",
)

function prepare_state!(
    M::AbstractManifold,
    strategy::MLEStrategy,
    projection_argument::S,
    distribution,
    supplementary_η,
) where {S}
    return prepare_state!(
        M,
        getstate(strategy),
        strategy,
        projection_argument,
        distribution,
        supplementary_η,
    )
end

Base.@kwdef struct MLEStrategyState{F,C,G}
    targetfn::F
    config::C
    tmpgrad::G
end

function Base.:(==)(a::MLEStrategyState, b::MLEStrategyState)::Bool
    return a.targetfn == b.targetgn && a.config == b.config && a.tmpgrad == b.tmpgrad
end

gettargetfn(state::MLEStrategyState) = state.targetfn
getconfig(state::MLEStrategyState) = state.config
gettmpgrad(state::MLEStrategyState) = state.tmpgrad

function prepare_state!(
    M::AbstractManifold,
    ::Nothing,
    strategy::MLEStrategy,
    samples,
    distribution,
    supplementary_η,
)
    _, sample_container = ExponentialFamily.check_logpdf(distribution, samples)

    # Our samples are fixed, thus we can precompute all the `sufficientstatistics` once
    sufficientstatistics = zeros(
        paramfloattype(distribution),
        length(getnaturalparameters(distribution)),
        length(samples),
    )

    J = size(sufficientstatistics, 1)

    foreach(enumerate(sample_container)) do (i, sample)
        sample_sufficientstatistics = __projection_fast_pack_parameters(
            ExponentialFamily.sufficientstatistics(distribution, sample),
        )
        @turbo warn_check_args = false for j = 1:J
            @inbounds sufficientstatistics[j, i] = sample_sufficientstatistics[j]
        end
    end

    targetfn = MLETargetFn(M, samples, sufficientstatistics)
    config = ForwardDiff.GradientConfig(targetfn, getnaturalparameters(distribution))
    tmpgrad = ForwardDiff.gradient(targetfn, getnaturalparameters(distribution), config)
    return MLEStrategyState(targetfn, config, tmpgrad)
end

function prepare_state!(
    M::AbstractManifold,
    state::MLEStrategyState,
    strategy::MLEStrategy,
    samples,
    distribution,
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
    ef = convert(ExponentialFamilyDistribution, fn.manifold, η)
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
    obj::CVICostGradientObjective,
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
    obj::CVICostGradientObjective,
    strategy::MLEStrategy,
    state::MLEStrategyState,
    X,
    η,
    _,
    _,
    inv_fisher,
)
    G = ForwardDiff.gradient!(gettmpgrad(state), gettargetfn(state), η, getconfig(state))
    X = mul!(X, inv_fisher, G)
    return X
end