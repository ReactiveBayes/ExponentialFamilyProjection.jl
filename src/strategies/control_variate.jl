using StableRNGs, Bumper, FillArrays

import Random: AbstractRNG
import BayesBase: InplaceLogpdf

"""
    ControlVariateStrategy(; kwargs...)

A strategy for gradient descent optimization and gradients computations that resembles the REINFORCE gradient estimator.

The following parameters are available:
* `nsamples = 2000`: The number of samples to use for estimates
* `buffer = Bumper.SlabBuffer()`: Advanced option; A buffer for temporary computations

!!! note
    This strategy requires a function as an argument for `project_to` and cannot project a collection of samples. Use `MLEStrategy` to project a collection of samples.
"""
Base.@kwdef struct ControlVariateStrategy{S, B, TL}
    nsamples::S = 2000
    buffer::B = Bumper.SlabBuffer()
    base_logpdf_type::Type{TL}  = InplaceLogpdf
end

get_nsamples(strategy::ControlVariateStrategy) = strategy.nsamples
get_buffer(strategy::ControlVariateStrategy) = strategy.buffer

function Base.:(==)(a::ControlVariateStrategy, b::ControlVariateStrategy)::Bool
    return get_nsamples(a) == get_nsamples(b) && get_buffer(a) == get_buffer(b)
end

preprocess_strategy_argument(strategy::ControlVariateStrategy{S,B,TL}, argument::Any) where {S,B,TL} =
    (strategy, convert(TL, argument))
preprocess_strategy_argument(::ControlVariateStrategy, argument::AbstractArray) = error(
    lazy"The `ControlVariateStrategy` requires the projection argument to be a callable object (e.g. `Function`). Got `$(typeof(argument))` instead.",
)

Base.@kwdef struct ControlVariateStrategyState{M,L,LB,F,G}
    samples::M
    logpdfs::L
    logbasemeasures::LB
    sufficientstatistics::F
    gradsamples::G
end

function Base.:(==)(a::ControlVariateStrategyState, b::ControlVariateStrategyState)::Bool
    return a.samples == b.samples &&
           a.logpdfs == b.logpdfs &&
           a.logbasemeasures == b.logbasemeasures &&
           a.sufficientstatistics == b.sufficientstatistics &&
           a.gradsamples == b.gradsamples
end

get_samples(state::ControlVariateStrategyState) = state.samples
get_logpdfs(state::ControlVariateStrategyState) = state.logpdfs
get_logbasemeasures(state::ControlVariateStrategyState) = state.logbasemeasures
get_sufficientstatistics(state::ControlVariateStrategyState) = state.sufficientstatistics
get_gradsamples(state::ControlVariateStrategyState) = state.gradsamples

function create_state!(
    strategy::ControlVariateStrategy,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    initial_ef,
    supplementary_η,
)

    # If the `state` saved in `ControlVariateStrategy` is `nothing`
    # we simply create new containers for the samples, logpdfs, etc.
    nsamples = get_nsamples(strategy)
    rng = getrng(parameters)
    samples = prepare_samples_container(rng, initial_ef, nsamples, supplementary_η)
    logpdfs = prepare_logpdfs_container(rng, initial_ef, nsamples, supplementary_η)
    logbasemeasures =
        prepare_logbasemeasures_container(rng, initial_ef, nsamples, supplementary_η)
    sufficientstatistics =
        prepare_sufficientstatistics_container(rng, initial_ef, nsamples, supplementary_η)
    gradsamples = prepare_gradsamples_container(rng, initial_ef, nsamples, supplementary_η)

    state = ControlVariateStrategyState(
        samples = samples,
        logpdfs = logpdfs,
        logbasemeasures = logbasemeasures,
        sufficientstatistics = sufficientstatistics,
        gradsamples = gradsamples,
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

# The following functions are used to prepare the containers for the samples, logpdfs, etc.
prepare_samples_container(rng, distribution, nsamples, supplementary_η) =
    rand(rng, distribution, nsamples)
prepare_logpdfs_container(rng, distribution, nsamples, supplementary_η) =
    zeros(paramfloattype(distribution), nsamples)
prepare_sufficientstatistics_container(rng, distribution, nsamples, supplementary_η) =
    zeros(
        paramfloattype(distribution),
        length(getnaturalparameters(distribution)),
        nsamples,
    )
prepare_gradsamples_container(rng, distribution, nsamples, supplementary_η) =
    prepare_sufficientstatistics_container(rng, distribution, nsamples, supplementary_η)
# `logbasemeasures` container is a bit different, if the basemeasure is known to be constant, the 
# `log` of it can be precomputed and stored in the `lazy` container without actually allocating any space
prepare_logbasemeasures_container(rng, distribution, nsamples, supplementary_η) =
    prepare_logbasemeasures_container(
        ExponentialFamily.isbasemeasureconstant(distribution),
        rng,
        distribution,
        nsamples,
        supplementary_η,
    )

# We use `Fill` from `FillArrays` to create a container with the same value repeated `nsamples` times
# It does not allocate any memory, just stores the value and the number of times it should be repeated
prepare_logbasemeasures_container(
    ::ConstantBaseMeasure,
    rng,
    distribution,
    nsamples,
    supplementary_η,
) = Fill(
    (1 - length(supplementary_η)) * logbasemeasure(distribution, rand(rng, distribution)),
    nsamples,
)
# If the basemeasure is not constant, we allocate the memory
prepare_logbasemeasures_container(
    ::NonConstantBaseMeasure,
    rng,
    distribution,
    nsamples,
    supplementary_η,
) = zeros(paramfloattype(distribution), nsamples)

function prepare_state!(
    ::ControlVariateStrategy{S,B,TL},
    state::ControlVariateStrategyState,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    current_ef,
    supplementary_η,
) where {S,B,TL}

    # We need to reset the RNG state every time we prepare the state
    # This is important not only for reproducibility, but also to ensure
    # that the gradient computation is stable
    Random.seed!(getrng(parameters), getseed(parameters))
    Random.rand!(getrng(parameters), current_ef, get_samples(state))

    _, sample_container = ExponentialFamily.check_logpdf(current_ef, get_samples(state))

    glogpartion = ExponentialFamily.gradlogpartition(current_ef)
    J = size(get_gradsamples(state), 1)

    inplace_projection_argument! = convert(TL, projection_argument)
    inplace_projection_argument!(get_logpdfs(state), sample_container)

    one_minus_n_of_supplementary = 1 - length(supplementary_η)

    nonconstantbasemeasure =
        ExponentialFamily.isbasemeasureconstant(current_ef) === NonConstantBaseMeasure()

    foreach(enumerate(sample_container)) do (i, sample)
        # if `basemeasure` is constant we assume that 
        # the `log` of it has been precomputed before
        if nonconstantbasemeasure
            @inbounds state.logbasemeasures[i] =
                one_minus_n_of_supplementary *
                ExponentialFamily.logbasemeasure(current_ef, sample)
        end

        sufficientstatistics = __projection_fast_pack_parameters(
            ExponentialFamily.sufficientstatistics(current_ef, sample),
        )

        @inbounds logpdf = state.logpdfs[i]
        for j = 1:J
            @inbounds state.sufficientstatistics[j, i] = sufficientstatistics[j]
            @inbounds state.gradsamples[j, i] =
                (-state.logbasemeasures[i] + logpdf) *
                (state.sufficientstatistics[j, i] - glogpartion[j])
        end
    end

    return state
end

function compute_cost(
    M::AbstractManifold,
    strategy::ControlVariateStrategy,
    state::ControlVariateStrategyState,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    return dot(gradlogpartition, η) - mean(state.logpdfs) - logpartition +
           mean(state.logbasemeasures)
end

function compute_gradient!(
    M::AbstractManifold,
    strategy::ControlVariateStrategy,
    state::ControlVariateStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    buffer = get_buffer(strategy)
    if isnothing(buffer)
        return control_variate_compute_gradient!(
            strategy,
            state,
            X,
            η,
            logpartition,
            gradlogpartition,
            inv_fisher,
        )
    else
        return control_variate_compute_gradient_buffered!(
            strategy,
            state,
            X,
            η,
            logpartition,
            gradlogpartition,
            inv_fisher,
        )
    end
end

function control_variate_compute_gradient!(
    strategy::ControlVariateStrategy,
    state::ControlVariateStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)

    cov_matrix = cov(state.sufficientstatistics', state.gradsamples')
    corr_matrix = cov_matrix * inv_fisher
    mean_sufficientstats = @view(mean(state.sufficientstatistics, dims = 2)[:, 1])
    mean_gradsamples = @view(mean(state.gradsamples, dims = 2)[:, 1])

    estimated_grad_vector =
        mean_gradsamples - corr_matrix * (mean_sufficientstats - gradlogpartition)
    X .= (η - inv_fisher * estimated_grad_vector)

    return X
end

function control_variate_compute_gradient_buffered!(
    strategy::ControlVariateStrategy,
    state::ControlVariateStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    # This code is a bit involved, more comments are added
    # The `@no_escape` macro simplifies writing non-allocating code, it allows 
    # to create intermediate buffers which will be freed immediatelly upon exiting the block 
    # uses the buffer from `get_buffer(obj)` so buffer must be relatively big
    buffer = get_buffer(strategy)
    @no_escape buffer begin

        # First we compute the `cov` between `state.sufficientstatistics'` and `state.gradsamples'`
        # The naive code would be simply `cov_matrix = cov(state.sufficientstatistics', state.gradsamples')`
        # but it allocates A LOT, especially when we have a lot of samples, so instead we preallocate the space 
        # using the `@alloc` macro and call inplace `control_variate_cov_buffered!`
        # --
        cov_matrix = @alloc(
            promote_type(eltype(state.sufficientstatistics), eltype(state.gradsamples)),
            size(state.sufficientstatistics, 1),
            size(state.gradsamples, 1)
        )
        control_variate_cov_buffered!(
            buffer,
            cov_matrix,
            state.sufficientstatistics',
            state.gradsamples',
        )

        # Compute means of sufficientstatistics and gradsamples inplace
        # The naive code would be 
        # `mean_sufficientstats = mean(cache.sufficientstatistics, dims = 2)[:, 1]`
        # `mean_gradsamples = mean(cache.gradsamples, dims = 2)[:, 1]`
        # --
        mean_sufficientstats =
            @alloc(eltype(state.sufficientstatistics), size(state.sufficientstatistics, 1))
        mean_gradsamples =
            @alloc(eltype(state.sufficientstatistics), size(state.gradsamples, 1))
        mean!(mean_sufficientstats, state.sufficientstatistics)
        mean!(mean_gradsamples, state.gradsamples)
        # --

        # The next four lines finish the computation, and essentially equivalent to the following code 
        # `estimated_grad_vector = mean_gradsamples - corr_matrix * (mean_sufficientstats - gradlogpartition)`
        # where `corr_matrix = cov_matrix * inv_fisher`
        # `ef_gradient = η - inv_fisher * estimated_grad_vector` # or (η - (η_ef + inv_fisher * estimated_grad_vector))
        # --
        tmp1 = @alloc(
            promote_type(eltype(mean_sufficientstats), eltype(gradlogpartition)),
            length(mean_sufficientstats)
        )
        tmp2 = @alloc(promote_type(eltype(inv_fisher), eltype(tmp1)), length(tmp1))
        tmp3 = @alloc(promote_type(eltype(cov_matrix), eltype(tmp2)), length(tmp2))

        map!(-, tmp1, mean_sufficientstats, gradlogpartition) # tmp1 = (mean_sufficientstats - gradlogpartition)
        mul!(tmp2, inv_fisher, tmp1) # tmp2 = inv_fisher * tmp1
        mul!(tmp3, cov_matrix, tmp2) # tmp3 = cov_matrix * tmp2, such that tmp3 = cov_matrix * inv_fisher * tmp1
        map!(-, tmp1, mean_gradsamples, tmp3)                 # tmp1 = estimated_grad_vector = mean_gradsamples - tmp3
        mul!(tmp3, inv_fisher, tmp1)                          # tmp3 = inv_fisher * estimated_grad_vector
        map!(-, X, η, tmp3)                                   # X .= η .- tmp3
        # --

        nothing
    end
    return X
end

function control_variate_cov_buffered!(buffer, Z, X, Y)

    @no_escape buffer begin
        _cov_tmp1 = @alloc(eltype(X), size(X, 2))
        _cov_tmp2 = @alloc(eltype(Y), size(Y, 2))
        _cov_tmp3 = @alloc(eltype(Z), size(X, 1), size(X, 2))
        _cov_tmp4 = @alloc(eltype(Z), size(Y, 1), size(Y, 2))

        BayesBase.mcov!(
            Z,
            X,
            Y,
            tmp1 = _cov_tmp1,
            tmp2 = _cov_tmp2,
            tmp3 = _cov_tmp3,
            tmp4 = _cov_tmp4,
        )

        nothing
    end
    return Z
end