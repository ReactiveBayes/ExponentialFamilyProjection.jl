"""
    GaussNewton{S,TL}

A deterministic strategy that resembles the Bonnet gradient with a point-mass approximation (no sampling).
For normal distributions, it evaluates logpdf/grad/Hessian once at the current mean and takes a step.
It implements the update akin to Eq. (13) in [Khan, 2024](https://arxiv.org/pdf/2107.04562).

!!! note
    This strategy requires a function as an argument for `project_to` and cannot project a collection of samples. Use `MLEStrategy` to project a collection of samples.
    This strategy requires a logpdf function that can be converted to an `InplaceLogpdfGradHess` object.
    This strategy requires the normal manifold.
"""
Base.@kwdef struct GaussNewton{S,TL}
    nsamples::S = 2000
    base_logpdf_type::Type{TL} = InplaceLogpdfGradHess
end

get_nsamples(strategy::GaussNewton) = strategy.nsamples

preprocess_strategy_argument(strategy::GaussNewton{S,TL}, argument::Any) where {S,TL} =
    (strategy, convert(TL, argument))
preprocess_strategy_argument(::GaussNewton, argument::AbstractArray) = error(
    lazy"The `GaussNewton` strategy requires the projection argument to be a callable object (e.g. `Function`) or an `InplaceLogpdfGradHess`. Got `$(typeof(argument))` instead.",
)

Base.@kwdef struct GaussNewtonState{S, L, LB, G, H, M}
    samples::S
    logpdfs::L
    logbasemeasures::LB
    grad::G
    hessian::H
    current_mean::M
end

# getters for compatibility
get_samples(state::GaussNewtonState) = state.samples
get_logpdfs(state::GaussNewtonState) = state.logpdfs
get_logbasemeasures(state::GaussNewtonState) = state.logbasemeasures

function create_state!(
    strategy::GaussNewton,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    initial_ef,
    supplementary_η,
)
    nsamples = get_nsamples(strategy)
    rng = getrng(parameters)
    xdim = length(mean(initial_ef))
    T = paramfloattype(initial_ef)
    state = GaussNewtonState(
        samples = prepare_samples_container(rng, initial_ef, nsamples, supplementary_η),
        logpdfs = prepare_logpdfs_container(rng, initial_ef, nsamples, supplementary_η),
        logbasemeasures = prepare_logbasemeasures_container(rng, initial_ef, nsamples, supplementary_η),
        grad = zeros(T, xdim),
        hessian = zeros(T, xdim, xdim),
        current_mean = zeros(T, xdim),
    )
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

function _compute_grad_hess_state!(::Any, state, inplace_projection_argument!)
    grad!(inplace_projection_argument!, state.grad, state.current_mean)
    hess!(inplace_projection_argument!, state.hessian, state.current_mean)
end

function _compute_grad_hess_state!(::Type{ExponentialFamily.NormalMeanVariance}, state, inplace_projection_argument!)
    grad!(inplace_projection_argument!, state.grad, state.current_mean[1])
    hess!(inplace_projection_argument!, state.hessian, state.current_mean[1])
end

function prepare_state!(
    ::GaussNewton{S,TL},
    state::GaussNewtonState,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    current_ef,
    supplementary_η,
) where {S,TL}
    inplace_projection_argument! = convert(TL, projection_argument)

    # Rerandomize samples for cost estimates
    Random.seed!(getrng(parameters), getseed(parameters))
    Random.rand!(getrng(parameters), current_ef, get_samples(state))

    _, sample_container = ExponentialFamily.check_logpdf(current_ef, get_samples(state))
    one_minus_n_of_supplementary = 1 - length(supplementary_η)
    nonconstantbasemeasure =
        ExponentialFamily.isbasemeasureconstant(current_ef) === ExponentialFamily.NonConstantBaseMeasure()

    # Evaluate logpdf (and base measure if needed) for each sample for the cost
    for (i, sample) in enumerate(sample_container)
        if nonconstantbasemeasure
            @inbounds get_logbasemeasures(state)[i] =
                one_minus_n_of_supplementary * ExponentialFamily.logbasemeasure(current_ef, sample)
        end
        logpdf!(inplace_projection_argument!, view(get_logpdfs(state), i:i), sample)
    end

    # Compute current mean from natural parameters: (-2η₂) m = η₁
    current_nat_param = getnaturalparameters(current_ef)
    typetag = ExponentialFamily.exponential_family_typetag(current_ef)
    η1, η2 = ExponentialFamily.unpack_parameters(typetag, current_nat_param)
    state.current_mean .= (-2η2) \ η1

    # Evaluate grad/hessian once at the current mean for deterministic gradient
    ef_typetag = ExponentialFamily.exponential_family_typetag(current_ef)
    _compute_grad_hess_state!(ef_typetag, state, inplace_projection_argument!)
    return state
end

function compute_cost(
    ::AbstractManifold,
    ::GaussNewton,
    state::GaussNewtonState,
    η,
    gradlogpartition,
    logpartition,
)
    return dot(gradlogpartition, η) - mean(get_logpdfs(state)) - logpartition + mean(get_logbasemeasures(state))
end

function compute_gradient!(
    M::AbstractManifold,
    ::GaussNewton,
    state::GaussNewtonState,
    X,
    η,
)
    grad_η1 = state.grad - state.hessian * state.current_mean
    grad_η2 = 0.5 * state.hessian
    typetag = ExponentialFamily.exponential_family_typetag(M)
    grad_vec = ExponentialFamily.pack_parameters(typetag, (grad_η1, grad_η2))
    X .= (η - grad_vec)
    return X
end

function call_objective(
    objective::ProjectionCostGradientObjective{J,F,C,P,S},
    M::AbstractManifold,
    X,
    p,
) where {J,F,C,P,S <: GaussNewton}
    current_ef = convert(ExponentialFamilyDistribution, M, p)
    current_η = copyto!(get_current_η(objective), getnaturalparameters(current_ef))

    strategy = get_strategy(objective)
    state = get_strategy_state(objective)
    projection_parameters = get_projection_parameters(objective)
    projection_argument = get_projection_argument(objective)
    supplementary_η = get_supplementary_η(objective)

    logpartition = ExponentialFamily.logpartition(current_ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(current_ef)

    state = prepare_state!(
        strategy,
        state,
        M,
        projection_parameters,
        projection_argument,
        current_ef,
        supplementary_η,
    )

    foreach(supplementary_η) do s_η
        map!(-, current_η, current_η, s_η)
    end

    c = compute_cost(
        M,
        strategy,
        state,
        current_η,
        gradlogpartition,
        logpartition,
    )

    X_nat = compute_gradient!(
        M,
        strategy,
        state,
        X,
        current_η,
    )
    X = jacobian_nat_to_manifold!(M, X, X_nat)
    X = project!(M, X, p, X)
    return c, X
end