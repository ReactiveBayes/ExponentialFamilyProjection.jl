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
    M::AbstractManifold,
    strategy::BonnetStrategy,
    state::BonnetStrategyState,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    return dot(gradlogpartition, η) - mean(state.logpdfs) - logpartition +
           mean(state.logbasemeasures)
end

function bonnet_compute_gradient!(
    M::AbstractManifold,
    ::BonnetStrategy,
    state::BonnetStrategyState,
    X,
    η,
    _,
    _,
    _,
)
    mean_grad_vector_η_1 = mean(state.grads, dims = 2)[:, 1]
    mean_hess_vector_η_2 = mean(state.hessians, dims = 3)[:, :, 1]
    grad_η1 = mean_grad_vector_η_1 - mean_hess_vector_η_2 * state.current_mean
    grad_η2 = 0.5 * mean_hess_vector_η_2
    typetag = ExponentialFamily.exponential_family_typetag(M)
    grad_vec = ExponentialFamily.pack_parameters(typetag, (grad_η1, grad_η2))
    X .= (η - grad_vec)
    return X
end
