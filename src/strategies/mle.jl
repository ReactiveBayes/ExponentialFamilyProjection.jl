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

function prepare_state!(
    M::AbstractManifold,
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
    targetfn = MLETargetFn(M, samples)
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

struct MLETargetFn{M,S}
    manifold::M
    samples::S
end

function (fn::MLETargetFn)(η)
    ef = convert(ExponentialFamilyDistribution, fn.manifold, η)
    _, container = ExponentialFamily.check_logpdf(ef, fn.samples)
    _logpartition = logpartition(ef)
    return -mean(
        x -> ExponentialFamily._plogpdf(ef, x, _logpartition, logbasemeasure(ef, x)),
        container,
    )
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