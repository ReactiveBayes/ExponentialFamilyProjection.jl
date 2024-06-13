using StableRNGs, LoopVectorization, Bumper

import BayesBase: InplaceLogpdf

"""
    ControlVariateStrategy(; kwargs...)

A strategy for gradient descent optimization and gradients computations.
The following parameters are available:
* `nsamples = 2000`: The number of samples to use for estimates
* `seed = 42`: The seed for the random number generator
* `rng = StableRNG(seed)`: The random number generator
"""
Base.@kwdef struct ControlVariateStrategy{S,D,N,T}
    nsamples::S = 2000
    seed::D = 42
    rng::N = StableRNG(seed)
    state::T = nothing
end

getnsamples(strategy::ControlVariateStrategy) = strategy.nsamples
getseed(strategy::ControlVariateStrategy) = strategy.seed
getrng(strategy::ControlVariateStrategy) = strategy.rng
getstate(strategy::ControlVariateStrategy) = strategy.state

function getinitialpoint(strategy::ControlVariateStrategy, M::AbstractManifold)
    return rand(getrng(strategy), M)
end

function with_state(strategy::ControlVariateStrategy, state)
    return ControlVariateStrategy(
        nsamples = getnsamples(strategy),
        seed = getseed(strategy),
        rng = getrng(strategy),
        state = state,
    )
end

function prepare_state!(
    strategy::ControlVariateStrategy,
    targetfn::F,
    distribution,
) where {F}
    return prepare_state!(
        getstate(strategy),
        strategy,
        convert(InplaceLogpdf, targetfn),
        distribution,
    )
end

Base.@kwdef struct ControlVariateStrategyState{M,L,F,G}
    samples::M
    logpdfs::L
    sufficientstatistics::F
    gradsamples::G
end

getsamples(state::ControlVariateStrategyState) = state.samples
getlogpdfs(state::ControlVariateStrategyState) = state.logpdfs
getsufficientstatistics(state::ControlVariateStrategyState) = state.sufficientstatistics
getgradsamples(state::ControlVariateStrategyState) = state.gradsamples

function prepare_state!(
    ::Nothing,
    strategy::ControlVariateStrategy,
    targetfn::InplaceLogpdf,
    distribution,
)

    # If the `state` saved in `ControlVariateStrategy` is `nothing`
    # we simply create new containers for the samples, logpdfs, etc.
    nsamples = getnsamples(strategy)
    rng = getrng(strategy)
    samples = rand(rng, distribution, nsamples)
    logpdfs = zeros(paramfloattype(distribution), nsamples)
    sufficientstatistics = zeros(
        paramfloattype(distribution),
        length(getnaturalparameters(distribution)),
        nsamples,
    )
    gradsamples = similar(sufficientstatistics)

    state = ControlVariateStrategyState(
        samples = samples,
        logpdfs = logpdfs,
        sufficientstatistics = sufficientstatistics,
        gradsamples = gradsamples,
    )

    return prepare_state!(state, strategy, targetfn, distribution)
end

function prepare_state!(
    state::ControlVariateStrategyState,
    strategy::ControlVariateStrategy,
    targetfn::InplaceLogpdf,
    distribution,
)

    # We need to reset the RNG state every time we prepare the state
    # This is important not only for reproducibility, but also to ensure
    # that the gradient computation is stable
    Random.seed!(getrng(strategy), getseed(strategy))
    Random.rand!(getrng(strategy), distribution, state.samples)

    _, sample_container = ExponentialFamily.check_logpdf(
        ExponentialFamily.variate_form(typeof(distribution)),
        typeof(state.samples),
        eltype(state.samples),
        distribution,
        state.samples,
    )

    glogpartion = ExponentialFamily.gradlogpartition(distribution)
    J = size(state.gradsamples, 1)

    targetfn(state.logpdfs, sample_container)

    foreach(enumerate(sample_container)) do (i, sample)
        @inbounds logpdf = state.logpdfs[i]

        sufficientstatistics = __control_variate_fast_pack_parameters(
            ExponentialFamily.sufficientstatistics(distribution, sample),
        )

        @turbo warn_check_args = false for j = 1:J
            @inbounds state.sufficientstatistics[j, i] = sufficientstatistics[j]
            @inbounds state.gradsamples[j, i] =
                logpdf * (state.sufficientstatistics[j, i] - glogpartion[j])
        end
    end

    return state
end

# This can go to the ExponentialFamily.jl, very useful
# The idea here is that it is not necessary to pack a tuple of numbers into a vector
__control_variate_fast_pack_parameters(t::NTuple{N,<:Number}) where {N} = t
__control_variate_fast_pack_parameters(t) = ExponentialFamily.pack_parameters(t)

function compute_cost(
    obj::CVICostGradientObjective,
    strategy::ControlVariateStrategy,
    state::ControlVariateStrategyState,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    trick = logsumexp(state.logpdfs) - log(strategy.nsamples)
    c = dot(gradlogpartition, η) + trick - logpartition
    return c
end

function compute_gradient!(
    obj::CVICostGradientObjective,
    strategy::ControlVariateStrategy,
    state::ControlVariateStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    buffer = get_cvi_buffer(obj)
    if isnothing(buffer)
        return control_variate_compute_gradient!(
            obj,
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
            obj,
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
    obj::CVICostGradientObjective,
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
    obj::CVICostGradientObjective,
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
    # uses the buffer from `get_cvi_buffer(obj)` so buffer must be relatively big
    buffer = get_cvi_buffer(obj)
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
        # --

        # Next we compute the `corr_matrix` using the sample principle, preallocate the storage
        # The naive code would be `corr_matrix = cov_matrix * inv_fisher`
        # --
        corr_matrix = @alloc(
            promote_type(eltype(cov_matrix), eltype(inv_fisher)),
            size(cov_matrix, 1),
            size(inv_fisher, 2)
        )
        mul!(corr_matrix, cov_matrix, inv_fisher)
        # --

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
        # `ef_gradient = η - inv_fisher * estimated_grad_vector` # or (η - (η_ef + inv_fisher * estimated_grad_vector))
        # --
        tmp1 = @alloc(
            promote_type(eltype(mean_sufficientstats), eltype(gradlogpartition)),
            length(mean_sufficientstats)
        )
        tmp2 = @alloc(promote_type(eltype(corr_matrix), eltype(tmp1)), length(tmp1))

        map!(-, tmp1, mean_sufficientstats, gradlogpartition) # tmp1 = (mean_sufficientstats - gradlogpartition)
        mul!(tmp2, corr_matrix, tmp1)                         # tmp2 = corr_matrix * tmp1
        map!(-, tmp1, mean_gradsamples, tmp2)                 # tmp1 = estimated_grad_vector = mean_gradsamples - tmp2
        mul!(tmp2, inv_fisher, tmp1)                          # tmp2 = inv_fisher * estimated_grad_vector
        map!(-, X, η, tmp2)                                   # X .= η .- tmp2
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