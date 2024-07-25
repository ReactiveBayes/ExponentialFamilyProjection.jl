using ForwardDiff

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

function prepare_state!(
    strategy::MLEStrategy,
    projection_argument::S,
    distribution,
    supplementary_η,
) where {S}
    if !isa(projection_argument, AbstractArray)
        error(
            lazy"`MLEStrategy` requires the projection argument to be an array of samples. Got `$(typeof(projection_argument))` instead.",
        )
    end
    return prepare_state!(
        getstate(strategy),
        strategy,
        projection_argument,
        distribution,
        supplementary_η,
    )
end

Base.@kwdef struct MLEStrategyState{S,G}
    samples::S
    tmpgrad::G
end

function Base.:(==)(a::MLEStrategyState, b::MLEStrategyState)::Bool
    return a.samples == b.samples && a.tmpgrad == b.tmpgrad
end

getsamples(state::MLEStrategyState) = state.samples
gettmpgrad(state::MLEStrategyState) = state.tmpgrad

function prepare_state!(
    ::Nothing,
    strategy::MLEStrategy,
    samples,
    distribution,
    supplementary_η,
)
    tmpgrad = similar(getnaturalparameters(distribution))
    return MLEStrategyState(samples, tmpgrad)
end

function prepare_state!(
    state::MLEStrategyState,
    strategy::MLEStrategy,
    samples,
    distribution,
    supplementary_η,
)
    if getsamples(state) != samples
        @warn "`MLEStrategyState` has a different set of samples. The projection algorithm may produce wrong results."
    end
    return state
end

function compute_cost(
    obj::CVICostGradientObjective,
    strategy::MLEStrategy,
    state::MLEStrategyState,
    η,
    _,
    _,
    _,
)
    ef = convert(ExponentialFamilyDistribution, get_cvi_manifold(obj), η)
    _, container = ExponentialFamily.check_logpdf(ef, getsamples(state))
    _logpartition = logpartition(ef)
    return -mean(
        x -> ExponentialFamily._plogpdf(ef, x, _logpartition, logbasemeasure(ef, x)),
        container,
    )
end

function compute_gradient!(
    obj::CVICostGradientObjective,
    strategy::MLEStrategy,
    state::MLEStrategyState,
    X,
    η,
    _,
    _,
    inv_fisher,
)
    f = (η) -> compute_cost(obj, strategy, state, η, nothing, nothing, nothing)
    G = ForwardDiff.gradient!(gettmpgrad(state), f, η)
    X = mul!(X, inv_fisher, G)

    return X
end