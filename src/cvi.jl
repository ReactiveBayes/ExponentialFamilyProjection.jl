using BayesBase, Random, StatsFuns, LinearAlgebra, FastCholesky, Bumper

struct CVICostGradientObjective{F,S}
    targetfn::F
    state::S
end

get_cvi_targetfn(obj::CVICostGradientObjective) = obj.targetfn
get_cvi_state(obj::CVICostGradientObjective) = obj.state

function (obj::CVICostGradientObjective)(M::AbstractManifold, X, p)
    ef = convert(ExponentialFamilyDistribution, M, p)

    state = get_cvi_state(obj)

    setup!(state, obj.targetfn, ef)

    logpartition = ExponentialFamily.logpartition(ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(ef)
    η = ExponentialFamily.getnaturalparameters(ef)
    inv_fisher = cholinv(ExponentialFamily.fisherinformation(ef))

    c = compute_cost(state, η, logpartition, gradlogpartition, inv_fisher)
    X = compute_gradient!(state, X, η, logpartition, gradlogpartition, inv_fisher)
    X = project!(M, X, p, X)

    return c, X
end

function compute_cost(state, η, logpartition, gradlogpartition, inv_fisher)
    # η_1 = ExponentialFamily.getnaturalparameters(cvi.exponetialfamilydistribution)
    trick = logsumexp(state.logpdfs) - log(state.nsamples)
    # return dot(ExponentialFamily.gradlogpartition(ef), nat_params - η_1) + trick - logpartition_part
    c = dot(gradlogpartition, η) + trick - logpartition
    return c
end

function compute_gradient!(state, X, η, logpartition, gradlogpartition, inv_fisher)
    # This code is a bit involved, more comments are added
    # The `@no_escape` macro simplifies writing non-allocating code, it allows 
    # to create intermediate buffers which will be freed immediatelly upon exiting the block 
    # uses the `state.buffer` so buffer must be relatively big
    @no_escape state.buffer begin

        # First we compute the `cov` between `state.sufficientstatistics'` and `state.gradsamples'`
        # The naive code would be simply `cov_matrix = cov(cache.sufficientstatistics', cache.gradsamples')`
        # but it allocates A LOT, especially when we have a lot of samples, so instead we preallocate the space 
        # using the `@alloc` macro and call inplace `cov!`
        # --
        cov_matrix = @alloc(
            promote_type(eltype(state.sufficientstatistics), eltype(state.gradsamples)),
            size(state.sufficientstatistics, 1),
            size(state.gradsamples, 1)
        )
        cov!(state, cov_matrix, state.sufficientstatistics', state.gradsamples')
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
        # `mean_vector = mean(cache.sufficientstatistics, dims = 2)[:, 1]`
        # `mean_gradsamples = mean(cache.gradsamples, dims = 2)`
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

using StableRNGs, LoopVectorization

Base.@kwdef struct CVIObjectiveState{T,S,L,F,G,B,D,N}
    nsamples::T
    samples::S
    logpdfs::L
    sufficientstatistics::F
    gradsamples::G
    buffer::B
    seed::D
    srng::N
end

function setup!(state::CVIObjectiveState, targetfn::InplaceLogpdf, ef)

    Random.seed!(state.srng, state.seed)
    Random.rand!(state.srng, ef, state.samples)

    _, sample_container = ExponentialFamily.check_logpdf(
        ExponentialFamily.variate_form(typeof(ef)),
        typeof(state.samples),
        eltype(state.samples),
        ef,
        state.samples,
    )

    glogpartion = ExponentialFamily.gradlogpartition(ef)
    J = size(state.gradsamples, 1)

    targetfn(state.logpdfs, sample_container)

    foreach(enumerate(sample_container)) do (i, sample)
        # logpdf = targetfn(sample)
        # @inbounds state.logpdfs[i] = logpdf
        @inbounds logpdf = state.logpdfs[i]

        # @inbounds state.gradsamples[:, i] .=
        #     (@view(state.sufficientstatistics[:, i]) .- glogpartion) .* state.logpdfs[i]
        # @inbounds state.sufficientstatistics[:, i] .=
        #     __fast_pack(ExponentialFamily.sufficientstatistics(ef, s))

        sufficientstatistics =
            __fast_pack_parameters(ExponentialFamily.sufficientstatistics(ef, sample))

        @turbo warn_check_args = false for j = 1:J
            @inbounds state.sufficientstatistics[j, i] = sufficientstatistics[j]
            @inbounds state.gradsamples[j, i] =
                logpdf * (state.sufficientstatistics[j, i] - glogpartion[j])
        end
    end

end

# This can go to the ExponentialFamily.jl, very useful
# The idea here is that it is not necessary to pack a tuple of numbers into a vector
__fast_pack_parameters(t::NTuple{N,<:Number}) where {N} = t
__fast_pack_parameters(t) = ExponentialFamily.pack_parameters(t)

"""
    cov!(cache::CVIObjectiveCache, Z, X, Y)

Computes `cov(X, Y)` where both `X` and `Y` are matrices and stores the result in `Z`. Uses intermediate storage from `buffer` in `state`.
"""
function cov!(state::CVIObjectiveState, Z, X, Y)

    @no_escape state.buffer begin
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