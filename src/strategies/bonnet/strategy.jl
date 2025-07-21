using StableRNGs

import Random: AbstractRNG

"""
    BonnetStrategy{S, TL}

A strategy for gradient descent optimization and gradients computations that resembles the Bonnet gradient estimator that works for normal distributions.
It's based on the equations (10) and (11) in [Khan, 2024](https://arxiv.org/pdf/2107.04562).

The following parameters are available:
* `nsamples = 2000`: The number of samples to use for estimates

!!! note
    This strategy requires a function as an argument for `project_to` and cannot project a collection of samples. Use `MLEStrategy` to project a collection of samples.
    This strategy requires a logpdf function that can be converted to an `InplaceLogpdfGradHess` object.
    This strategy requires the normal manifold.
"""
Base.@kwdef struct BonnetStrategy{S, TL}
    nsamples::S = 2000
    base_logpdf_type::Type{TL} = InplaceLogpdfGradHess
end

get_nsamples(strategy::BonnetStrategy) = strategy.nsamples

preprocess_strategy_argument(strategy::BonnetStrategy{S,TL}, argument::Any) where {S,TL} =
    (strategy, convert(TL, argument))
preprocess_strategy_argument(::BonnetStrategy, argument::AbstractArray) = error(
    lazy"The `BonnetStrategy` requires the projection argument to be a callable object (e.g. `Function`) or an `InplaceLogpdfGradHess`. Got `$(typeof(argument))` instead.",
)

Base.@kwdef struct BonnetStrategyState{S, L, G, H, M}
    samples::S
    logpdfs::L
    grads::G
    hessians::H
    current_mean::M
end

# Getter functions for BonnetStrategyState
get_samples(state::BonnetStrategyState) = state.samples
get_logpdfs(state::BonnetStrategyState) = state.logpdfs
get_grads(state::BonnetStrategyState) = state.grads
get_hessians(state::BonnetStrategyState) = state.hessians
get_current_mean(state::BonnetStrategyState) = state.current_mean

function create_state!(
    strategy::BonnetStrategy,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    initial_ef,
    supplementary_η,
)
    # Create containers for the BonnetStrategy state
    nsamples = get_nsamples(strategy)
    rng = getrng(parameters)
    
    # Prepare containers following the same pattern as ControlVariateStrategy
    samples = prepare_samples_container(rng, initial_ef, nsamples, supplementary_η)
    logpdfs = prepare_logpdfs_container(rng, initial_ef, nsamples, supplementary_η)
    grads = prepare_grads_container(rng, initial_ef, nsamples, supplementary_η)
    hessians = prepare_hessians_container(rng, initial_ef, nsamples, supplementary_η)
    current_mean = prepare_current_mean_container(rng, initial_ef, supplementary_η)

    state = BonnetStrategyState(
        samples = samples,
        logpdfs = logpdfs,
        grads = grads,
        hessians = hessians,
        current_mean = current_mean,
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

# Helper functions to prepare containers for BonnetStrategy
prepare_samples_container(rng, distribution, nsamples, supplementary_η) =
    rand(rng, distribution, nsamples)
prepare_logpdfs_container(rng, distribution, nsamples, supplementary_η) =
    zeros(paramfloattype(distribution), nsamples)
prepare_grads_container(rng, distribution, nsamples, supplementary_η) =
    zeros(
        paramfloattype(distribution),
        length(mean(distribution)),  # dimension of the sample space
        nsamples,
    )
prepare_hessians_container(rng, distribution, nsamples, supplementary_η) =
    zeros(
        paramfloattype(distribution),
        length(mean(distribution)),  # dimension of the sample space
        length(mean(distribution)),  # dimension of the sample space  
        nsamples,
    )
prepare_current_mean_container(rng, distribution, supplementary_η) =
    zeros(paramfloattype(distribution), length(mean(distribution)))

function prepare_state!(
    ::BonnetStrategy{S,TL},
    state::BonnetStrategyState,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    current_ef,
    supplementary_η,
) where {S,TL}

    # We need to reset the RNG state every time we prepare the state
    # This is important not only for reproducibility, but also to ensure
    # that the gradient computation is stable
    Random.seed!(getrng(parameters), getseed(parameters))
    Random.rand!(getrng(parameters), current_ef, get_samples(state))

   
    _, sample_container = ExponentialFamily.check_logpdf(current_ef, get_samples(state))
    inplace_projection_argument! = convert(TL, projection_argument)
    
    # Evaluate logpdf, grad, and hess for each sample
    for (i, sample) in enumerate(sample_container)
        logpdf!(inplace_projection_argument!, view(get_logpdfs(state), i:i), sample)
        grad!(inplace_projection_argument!, view(get_grads(state), :, i), sample)
        hess!(inplace_projection_argument!, view(get_hessians(state), :, :, i), sample)
    end
    
    current_nat_param = getnaturalparameters(current_ef)
    exponential_family_typetag = ExponentialFamily.exponential_family_typetag(current_ef)
    η1, η2 = ExponentialFamily.unpack_parameters(exponential_family_typetag, current_nat_param)
    state.current_mean .= (-2η2) \ η1
    return state
end

function compute_cost(
    ::AbstractManifold,
    ::BonnetStrategy,
    state::BonnetStrategyState,
    η,
    gradlogpartition,
)
    return dot(gradlogpartition, η) - mean(state.logpdfs) - logpartition +
           mean(state.logbasemeasures)
end

function compute_gradient!(
    M::AbstractManifold,
    strategy::BonnetStrategy,
    state::BonnetStrategyState,
    X,
    η,
)
    return bonnet_compute_gradient!(M, strategy, state, X, η)
end

function bonnet_compute_gradient!(
    M::AbstractManifold,
    ::BonnetStrategy,
    state::BonnetStrategyState,
    X,
    η
)
    mean_grad_vector_η_1 = mean(get_grads(state), dims = 2)[:, 1]
    mean_hess_vector_η_2 = mean(get_hessians(state), dims = 3)[:, :, 1]
    grad_η1 = mean_grad_vector_η_1 - mean_hess_vector_η_2 * state.current_mean
    grad_η2 = 0.5 * mean_hess_vector_η_2
    typetag = ExponentialFamily.exponential_family_typetag(M)
    grad_vec = ExponentialFamily.pack_parameters(typetag, (grad_η1, grad_η2))
    X .= (η - grad_vec)
    return X
end

function call_objective(
    objective::ProjectionCostGradientObjective{J,F,C,P,S},
    M::AbstractManifold,
    X,
    p
) where {J,F,C,P,S <: BonnetStrategy}
    current_ef = convert(ExponentialFamilyDistribution, M, p)
    current_η = copyto!(get_current_η(objective), getnaturalparameters(current_ef))

    strategy = get_strategy(objective)
    state = get_strategy_state(objective)
    projection_parameters = get_projection_parameters(objective)
    projection_argument = get_projection_argument(objective)
    supplementary_η = get_supplementary_η(objective)

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
        gradlogpartition
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


