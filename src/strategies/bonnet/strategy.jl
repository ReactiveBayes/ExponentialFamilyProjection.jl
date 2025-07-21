using StableRNGs, FillArrays, StaticTools

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
    
    current_nat_param = getnatparam(current_ef)
    η_1 = view(current_nat_param, 1:dim_size)
    η_2 = view(current_nat_param, dim_size+1:length(current_nat_param))
    state.current_mean .= (-2η_2) \ η_1
    return state
end
