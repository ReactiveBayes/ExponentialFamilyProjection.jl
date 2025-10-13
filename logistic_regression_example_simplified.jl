using LinearAlgebra
using Random
using StableRNGs
using Distributions
using ExponentialFamily
using ExponentialFamilyProjection
using Plots
using PythonCall

let
    rng = StableRNG(42)

    n = 600                 # number of observations
    input_dim = 2          # 2D features for visualization
    d = input_dim + 1       # add intercept term

    # Build design matrix with intercept as first column
    X_feat = randn(rng, n, input_dim)
    X = hcat(ones(n), X_feat)
    β_true = [0.5, 2.0, -1.5]  # intercept, weights for x1, x2
    logits = X * β_true
    σ(z) = 1.0 / (1.0 + exp(-z))
    p = map(σ, logits)
    y = rand.(Ref(rng), Bernoulli.(p))

    # Define in-place log posterior, gradient, Hessian for β
    # log likelihood contribution: sum(y_i * x_i'β - log(1 + exp(x_i'β)))
    # grad: X' (y - σ(Xβ))
    # hess: - X' W X, where W = Diag(σ(Xβ) * (1 - σ(Xβ)))
    function logpost!(out::AbstractVector{T}, β::AbstractVector{T}) where {T<:Real}
        Xβ = X * β
        # Use numerically stable log(1 + exp(z))
        function log1pexp(z)
            if z > 0
                return z + log1p(exp(-z))
            else
                return log1p(exp(z))
            end
        end
        s = zero(T)
        @inbounds for i in 1:n
            s += y[i] * Xβ[i] - log1pexp(Xβ[i])
        end
        s += -0.5 * dot(β, β)
        out[1] = s
        return out
    end

    function grad!(out::AbstractVector{T}, β::AbstractVector{T}) where {T<:Real}
        fill!(out, 0)
        Xβ = X * β
        @inbounds for i in 1:n
            # σ(Xβ[i])
            pi = 1.0 / (1.0 + exp(-Xβ[i]))
            # contribution to gradient: (y[i] - pi) * x_i
            @views out[:] .+= (y[i] - pi) .* X[i, :]
        end
        return out
    end

    function hess!(out::AbstractMatrix{T}, β::AbstractVector{T}) where {T<:Real}
        Xβ = X * β
        # W diagonal entries: pi * (1 - pi)
        # We build X' * W * X efficiently
        fill!(out, 0)
        @inbounds for i in 1:n
            pi = 1.0 / (1.0 + exp(-Xβ[i]))
            wi = pi * (1 - pi)
            # rank-1 update: out -= wi * (x_i x_i')
            @views out .-= wi .* (X[i, :] * transpose(X[i, :]))
        end
        return out
    end

    inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(logpost!, grad!, hess!)

    # We project the (unnormalized) posterior to a multivariate normal in natural coordinates
    params = ProjectionParameters(
        tolerance = 1e-8,
        strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 1), # deterministic, no sampling used for grad/hess
    )

    prj = ProjectedTo(MvNormalMeanCovariance, d, parameters = params)
    ef = project_to(prj, inplace)

    μ_est = mean(ef)
    Σ_est = cov(ef)
    # Visualization: decision regions in 2D feature space
    x1_min = minimum(X[:, 2]) - 3.0
    x1_max = maximum(X[:, 2]) + 3.0
    x2_min = minimum(X[:, 3]) - 3.0
    x2_max = maximum(X[:, 3]) + 3.0

    xs = range(x1_min, x1_max; length = 200)
    ys = range(x2_min, x2_max; length = 200)
    Z = Array{Float64}(undef, length(xs), length(ys))
    for (i, x1) in enumerate(xs)
        for (j, x2) in enumerate(ys)
            z = μ_est[1] + μ_est[2] * x1 + μ_est[3] * x2
            Z[i, j] = 1.0 / (1.0 + exp(-z))
        end
    end

    # Mean-based binary decision regions (two-color background)
    Zbin = ifelse.(Z .>= 0.5, 1.0, 0.0)
    palette_regions = cgrad([:lightcoral, :lightgreen], 2, categorical = true)
    plt_mean = heatmap(xs, ys, Zbin'; c = palette_regions, alpha = 0.35, colorbar = false)
    contour!(xs, ys, Z'; levels = [0.5], linecolor = :black, linewidth = 2.5, label = nothing)
    scatter!(X[y .== 0, 2], X[y .== 0, 3]; markersize = 5.5, markerstrokecolor = :white, markerstrokewidth = 0.8, label = "y = 0", color = :red4)
    scatter!(X[y .== 1, 2], X[y .== 1, 3]; markersize = 5.5, markerstrokecolor = :white, markerstrokewidth = 0.8, label = "y = 1", color = :green4)
    xlabel!("x₁")
    ylabel!("x₂")
    title!("Posterior mean decision regions")
    savefig(plt_mean, "logistic_regression_classification.png")
    println("Saved figure: logistic_regression_classification.png")

    # Monte Carlo predictive map by sampling β ~ N(μ_est, Σ_est)
    nsamples_pred = 200
    Zmc = zeros(length(xs), length(ys))
    mvn_post = MvNormal(μ_est, Symmetric(Σ_est))
    for s in 1:nsamples_pred
        βs = rand(mvn_post)
        for (i, x1) in enumerate(xs)
            for (j, x2) in enumerate(ys)
                z = βs[1] + βs[2] * x1 + βs[3] * x2
                Zmc[i, j] += 1.0 / (1.0 + exp(-z))
            end
        end
    end
    Zmc ./= nsamples_pred

    # MC-averaged predictive map with red→green gradient (0 to 1)
    plt_mc = contourf(xs, ys, Zmc'; levels = 0:0.05:1, c = cgrad([:red, :green]), alpha = 0.6, colorbar_title = "E[P(y=1)]", contour_lines = false, linecolor = :transparent, linewidth = 0)
    contour!(xs, ys, Zmc'; levels = [0.5], linecolor = :black, linewidth = 2.5, label = nothing)
    scatter!(X[y .== 0, 2], X[y .== 0, 3]; markersize = 5.5, markerstrokecolor = :white, markerstrokewidth = 0.8, label = "y = 0", color = :red4)
    scatter!(X[y .== 1, 2], X[y .== 1, 3]; markersize = 5.5, markerstrokecolor = :white, markerstrokewidth = 0.8, label = "y = 1", color = :green4)
    xlabel!("x₁")
    ylabel!("x₂")
    title!("MC-averaged predictive map")
    savefig(plt_mc, "logistic_regression_classification_mc.png")
    println("Saved figure: logistic_regression_classification_mc.png")

    # Side-by-side comparison figure
    plt_grid = plot(plt_mean, plt_mc; layout = (1, 2), size = (1000, 400))
    savefig(plt_grid, "logistic_regression_classification_compare.png")
    println("Saved figure: logistic_regression_classification_compare.png")

end


