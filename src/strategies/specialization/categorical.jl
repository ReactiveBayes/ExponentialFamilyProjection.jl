# Categorical stores K logits with the last fixed to zero. Its full K × K Fisher
# matrix is singular, so work in the first K-1 independent coordinates instead.
prepare_inv_fisher(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    ::Union{ControlVariateStrategy,MLEStrategy},
    current_ef,
) where {F} = nothing

"""
    categorical_inv_fisher_mul!(out, p, v)

Apply the inverse categorical Fisher matrix in reference-category coordinates.
`p` contains all K positive probabilities; `out` and `v` have K-1 entries.
For `F = Diagonal(p[1:end-1]) - p[1:end-1] * p[1:end-1]'`, the inverse
action is `v ./ p[1:end-1] .+ sum(v) / p[end]`. `out` may alias `v`.
"""
function categorical_inv_fisher_mul!(out, p, v)
    correction = sum(v) / p[end]
    for i in eachindex(v)
        out[i] = v[i] / p[i] + correction
    end
    return out
end

function compute_gradient!(
    ::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    strategy::ControlVariateStrategy,
    state::ControlVariateStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
) where {F}
    # Keep the sampled estimator, restricting both its statistics and gradients
    # to independent coordinates. No additional target evaluations are needed.
    n = length(gradlogpartition) - 1
    sufficientstatistics = @view state.sufficientstatistics[1:n, :]
    gradsamples = @view state.gradsamples[1:n, :]
    cov_matrix = cov(sufficientstatistics', gradsamples')
    mean_sufficientstats = vec(mean(sufficientstatistics, dims = 2))
    mean_gradsamples = vec(mean(gradsamples, dims = 2))

    correction = mean_sufficientstats - view(gradlogpartition, 1:n)
    categorical_inv_fisher_mul!(correction, gradlogpartition, correction)
    estimated_grad = mean_gradsamples - cov_matrix * correction
    categorical_inv_fisher_mul!(estimated_grad, gradlogpartition, estimated_grad)

    for i = 1:n
        X[i] = η[i] - estimated_grad[i]
    end
    X[end] = zero(eltype(X))
    return X
end

function compute_gradient!(
    M::ExponentialFamilyManifolds.NaturalParametersManifold{F,Categorical},
    strategy::MLEStrategy,
    state::MLEStrategyState,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
) where {F}
    # Match the generic MLE objective, including a possible supplementary shift
    # in η. The Fisher metric is evaluated at the original current distribution,
    # whose probabilities are supplied in gradlogpartition.
    ef = convert(
        ExponentialFamilyDistribution,
        M,
        ExponentialFamilyManifolds.partition_point(M, η),
    )
    mean_sufficientstats = vec(mean(gettargetfn(state).sufficientstatistics, dims = 2))
    gradient = ExponentialFamily.gradlogpartition(ef) - mean_sufficientstats
    n = length(gradlogpartition) - 1
    categorical_inv_fisher_mul!(view(X, 1:n), gradlogpartition, view(gradient, 1:n))
    X[end] = zero(eltype(X))
    return X
end
