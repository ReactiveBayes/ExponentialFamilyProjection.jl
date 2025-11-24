```@meta
CurrentModule = ExponentialFamilyProjection
```

# ExponentialFamilyProjection

The `ExponentialFamilyProjection.jl` package offers a suite of functions for projecting an arbitrary (un-normalized) log probability density function onto a specified member of the exponential family (e.g., Gaussian, Beta, Bernoulli). This is achieved by optimizing the natural parameters of the exponential family member within a defined manifold. The library leverages `Manopt.jl` for optimization and utilizes [`ExponentialFamilyManifolds.jl`](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl) to define the manifolds corresponding to the members of the exponential family.

## Projection parameters

In order to project a log probability density function onto a member of the exponential family, the user first needs to specify projection parameters:

```@docs
ExponentialFamilyProjection.ProjectionParameters
ExponentialFamilyProjection.DefaultProjectionParameters
ExponentialFamilyProjection.getinitialpoint
```

Read more about different optimization strategies [here](@ref opt-strategies).

## Projection family

After the parameters have been specified the user can proceed with specifying the projection type (exponential family member), its dimensionality and (optionally) the conditioner.

```@docs
ExponentialFamilyProjection.ProjectedTo
```

## Projection 

The projection is performed by calling the `project_to` function with the specified [`ExponentialFamilyProjection.ProjectedTo`](@ref) and log probability density function or a set of data point as the second argument.

```@docs 
ExponentialFamilyProjection.project_to
```

## [Optimization strategies](@id opt-strategies)

The optimization procedure requires computing the expectation of the gradient to perform gradient descent in the natural parameters space. Currently, the library provides the following strategies for computing these expectations:

```@docs
ExponentialFamilyProjection.DefaultStrategy
ExponentialFamilyProjection.ControlVariateStrategy
ExponentialFamilyProjection.ClosedFormStrategy
ExponentialFamilyProjection.MLEStrategy
ExponentialFamilyProjection.BonnetStrategy
ExponentialFamilyProjection.GaussNewton
ExponentialFamilyProjection.preprocess_strategy_argument
ExponentialFamilyProjection.create_state!
ExponentialFamilyProjection.prepare_state!
ExponentialFamilyProjection.compute_cost
ExponentialFamilyProjection.compute_gradient!
```

## In-place logpdf/grad/Hessian adapters

The library provides convenient wrappers to evaluate log-density, gradient, and Hessian in-place, and an adapter to combine separate `grad!`/`hess!` into a single `grad_hess!`.

```@docs
ExponentialFamilyProjection.InplaceLogpdfGradHess
ExponentialFamilyProjection.InplaceLogpdfGradHess(::Any, ::Any, ::Any)
ExponentialFamilyProjection.NaiveGradHess
ExponentialFamilyProjection.logpdf!(::ExponentialFamilyProjection.InplaceLogpdfGradHess, ::Any, ::Any)
ExponentialFamilyProjection.grad_hess!(::ExponentialFamilyProjection.InplaceLogpdfGradHess, ::Any, ::Any, ::Any)
ExponentialFamilyProjection.grad_hess!(::ExponentialFamilyProjection.NaiveGradHess, ::Any, ::Any, ::Any)
```

For high-dimensional distributions, adjusting the default number of samples might be necessary to achieve better performance.

## Examples

### Gaussian projection

In this example we project an arbitrary log probability density function onto a Gaussian distribution. The log probability density function is defined using another `Gaussian`, but it can be any function:

```@example projection
using ExponentialFamilyProjection, ExponentialFamily, BayesBase
using Test #hide
using Distributions #hide

hiddengaussian = NormalMeanVariance(3.14, 2.71)
targetf = (x) -> logpdf(hiddengaussian, x)
prj = ProjectedTo(NormalMeanVariance)
result = project_to(prj, targetf)
@test kldivergence(result, hiddengaussian) < 1e-3 #hide
result #hide
```

We can see that the estimated `result` is pretty close to the actual `hiddengaussian` used to define the `targetf`. We can also visualise the results using the `Plots.jl` package.

```@example projection
using Plots

plot(-6.0:0.1:12.0, x -> pdf(hiddengaussian, x), label="real distribution", fill = 0, fillalpha = 0.2)
plot!(-6.0:0.1:12.0, x -> pdf(result, x), label="estimated projection", fill = 0, fillalpha = 0.2)
```

Let's also try to project an arbitrary unnormalized log probability density function onto a Gaussian distribution:

```@example projection
# `+ 100` to ensure that the function is unnormalized
targetf = (x) -> -0.5 * (x - 3.14)^2 + 100

result = project_to(prj, targetf)
```

In this case, `targetf` does not define any valid probability distribution since it is unnormalized, but the `project_to` function is able to project it onto a closest possible Gaussian distribution. We can again visualize the results using the `Plots.jl` package:

```@example projection
plot(-40.0:0.1:40.0, targetf, label="unnormalized logpdf", fill = 0, fillalpha = 0.2)
plot!(-40.0:0.1:40.0, (x) -> logpdf(result, x), label="estimated logpdf of a Gaussian", fill = 0, fillalpha = 0.2)
```

### Beta projection

The experiment can be performed for other members of the exponential family as well. For example, let's project an arbitrary log probability density function onto a Beta distribution:

```@example projection
hiddenbeta = Beta(10, 3)
targetf = (x) -> logpdf(hiddenbeta, x)
prj = ProjectedTo(Beta)
result = project_to(prj, targetf)
@test kldivergence(result, hiddenbeta) < 1e-2 #hide
result #hide
```

And let's visualize the result using the `Plots.jl` package:

```@example projection
plot(0.0:0.01:1.0, x -> pdf(hiddenbeta, x), label="real distribution", fill = 0, fillalpha = 0.2)
plot!(0.0:0.01:1.0, x -> pdf(result, x), label="estimated projection", fill = 0, fillalpha = 0.2)
```

### Multivariate Gaussian projection

The library also supports multivariate distributions. Let's project an arbitrary log probability density function onto a multivariate Gaussian distribution.

```@example projection
hiddengaussian = MvNormalMeanCovariance(
    [ 3.14, 2.17 ],
    [ 2.0 -0.1; -0.1 3.0 ]
)
targetf = (x) -> logpdf(hiddengaussian, x)
prj = ProjectedTo(MvNormalMeanCovariance, 2)
result = project_to(prj, targetf)
@test kldivergence(result, hiddengaussian) < 1e-2 #hide
result #hide
```

As in previous examples the result is pretty close to the actual `hiddengaussian` used to define the `targetf`. 

### Gauss–Newton strategy (logistic regression)

The Gauss–Newton strategy uses first and second derivatives of the target log-density to form a deterministic update, avoiding Monte Carlo sampling. This is useful when you can provide in-place `logpdf!`, `grad!`, and `hess!` for your target. Below we demonstrate projecting a Bayesian logistic regression model (which is not a normalized distribution) onto a multivariate Gaussian using Gauss–Newton strategy `GaussNewton`.

We split this example into small steps and use a shared example environment so that variables (including a stable RNG) persist between blocks.

In the following block we sample `X` (our features) and `y` (binary outputs).

```@example gaussnewton
using LinearAlgebra
using StableRNGs
using Distributions
using ExponentialFamily
using ExponentialFamilyProjection
using StatsFuns
using Plots

# 1) Generate a reproducible dataset (shared RNG)
rng = StableRNG(42)
n = 600
input_dim = 2
d = input_dim + 1
X_feat = randn(rng, n, input_dim)
X = hcat(ones(n), X_feat)
β_true = [0.5, 2.0, -1.5]
p = map(logistic, X * β_true)
y = rand.(Ref(rng), Bernoulli.(p));
nothing # hide
```

We created a binary logistic regression dataset with an intercept and fixed `rng` for reproducibility.

```@example gaussnewton
# 2) Define in-place log-posterior, gradient, and Hessian
function logpost!(out::AbstractVector{T}, β::AbstractVector{T}) where {T<:Real}
    Xβ = X * β
    out[1] = mean(y .* Xβ .- log.(1 .+ exp.(Xβ)))
    return out
end

function grad!(out::AbstractVector{T}, β::AbstractVector{T}) where {T<:Real}
    fill!(out, 0)
    Xβ = X * β
    @inbounds for i in 1:n
        pi = 1 / (1 + exp(-Xβ[i]))
        @views out[:] .+= (y[i] - pi) .* X[i, :]
    end
    out .= out ./ length(y)
    return 
end

function hess!(out::AbstractMatrix{T}, β::AbstractVector{T}) where {T<:Real}
    Xβ = X * β
    fill!(out, 0)
    @inbounds for i in 1:n
        pi = 1 / (1 + exp(-Xβ[i]))
        wi = pi * (1 - pi)
        @views out .-= wi .* (X[i, :] * transpose(X[i, :]))
    end
    out .= out ./ length(y)
    return out
end
```

These in-place routines allow Gauss–Newton to form deterministic updates without Monte Carlo sampling.

```@example gaussnewton
# 3) Wrap and run Gauss–Newton projection
inplace = ExponentialFamilyProjection.InplaceLogpdfGradHess(logpost!, grad!, hess!)
params = ProjectionParameters(
    tolerance = 1e-8,
    strategy = ExponentialFamilyProjection.GaussNewton(nsamples = 1), # deterministic
)
prj = ProjectedTo(MvNormalMeanCovariance, d; parameters = params)
result = project_to(prj, inplace)
```

This projects the posterior to an `MvNormalMeanCovariance` parameterization using Gauss–Newton updates.

```@example gaussnewton
# 4) Inspect the projection result
μ = mean(result)
Σ = cov(result)
μ, size(Σ) # hide
```

Now we visualize the posterior-mean decision boundary and probability map. We compute a grid over feature space and evaluate the mean prediction σ(μ₀ + μ₁ x₁ + μ₂ x₂).

```@example gaussnewton
# 5) Build grid and compute posterior-mean probabilities
x1_min = minimum(X[:, 2]) - 3.0
x1_max = maximum(X[:, 2]) + 3.0
x2_min = minimum(X[:, 3]) - 3.0
x2_max = maximum(X[:, 3]) + 3.0

xs = range(x1_min, x1_max; length = 200)
ys = range(x2_min, x2_max; length = 200)
Z = Array{Float64}(undef, length(xs), length(ys))
for (i, x1) in enumerate(xs)
    for (j, x2) in enumerate(ys)
        z = μ[1] + μ[2] * x1 + μ[3] * x2
        Z[i, j] = 1.0 / (1.0 + exp(-z))
    end
end
```

```@example gaussnewton
# 6) Render probability heatmap and 0.5 decision contour with data overlay
plt_mean = contourf(
    xs, ys, Z';
    levels = 0:0.05:1,
    c = cgrad([:red, :green]),
    alpha = 0.65,
    colorbar_title = "P(y=1)",
    contour_lines = false,
    linecolor = :transparent,
    linewidth = 0,
    size = (650, 500),
)
contour!(xs, ys, Z'; levels = [0.5], linecolor = :black, linewidth = 3, label = nothing)
scatter!(
    X[y .== 0, 2], X[y .== 0, 3];
    markersize = 6,
    markerstrokecolor = :white,
    markerstrokewidth = 0.8,
    label = "y = 0",
    color = :red4,
)
scatter!(
    X[y .== 1, 2], X[y .== 1, 3];
    markersize = 6,
    markerstrokecolor = :white,
    markerstrokewidth = 0.8,
    label = "y = 1",
    color = :green4,
)
xlabel!("x₁")
ylabel!("x₂")
title!("mean boundary")
plt_mean # hide
```

To account for parameter uncertainty, we can estimate the predictive probability by Monte Carlo: sample coefficients β from the Gaussian posterior `result ~ N(μ, Σ)` and average σ(β₀ + β₁ x₁ + β₂ x₂) over samples. This yields a boundary reflecting posterior spread.

```@example gaussnewton
# 7) Monte Carlo-averaged predictive map from posterior β ~ N(μ, Σ)
nsamples_pred = 200
Zmc = zeros(length(xs), length(ys))
mvn_post = MvNormal(μ, Symmetric(Σ))
for s in 1:nsamples_pred
    βs = rand(rng, mvn_post)
    for (i, x1) in enumerate(xs)
        for (j, x2) in enumerate(ys)
            z = βs[1] + βs[2] * x1 + βs[3] * x2
            Zmc[i, j] += 1.0 / (1.0 + exp(-z))
        end
    end
end
Zmc ./= nsamples_pred
nothing # hide
```

```@example gaussnewton
# 8) Render MC-averaged probability heatmap and decision contour
plt_mc = contourf(
    xs, ys, Zmc';
    levels = 0:0.05:1,
    c = cgrad([:red, :green]),
    alpha = 0.65,
    colorbar_title = "E[P(y=1)]",
    contour_lines = false,
    linecolor = :transparent,
    linewidth = 0,
    size = (650, 500),
)
contour!(xs, ys, Zmc'; levels = [0.5], linecolor = :black, linewidth = 3, label = nothing)
scatter!(
    X[y .== 0, 2], X[y .== 0, 3];
    markersize = 6,
    markerstrokecolor = :white,
    markerstrokewidth = 0.8,
    label = "y = 0",
    color = :red4,
)
scatter!(
    X[y .== 1, 2], X[y .== 1, 3];
    markersize = 6,
    markerstrokecolor = :white,
    markerstrokewidth = 0.8,
    label = "y = 1",
    color = :green4,
)
xlabel!("x₁")
ylabel!("x₂")
title!("full posterior boundary")
plt_mc
```

```@example gaussnewton
# 9) Optional: side-by-side comparison
plot(plt_mean, plt_mc; layout = (1, 2), size = (1100, 450))
```

### How to use autograd with Gauss–Newton (Enzyme.jl)

You do not need to hand-derive gradients or Hessians to use Gauss–Newton. With `Enzyme.jl`, you can automatically obtain both and use them through the same in-place API shown above. In practice, this is typically faster and yields more stable estimates than naïve manual derivatives. `Enzyme.jl` has some sharp edges; please consult the [Enzyme documentation](https://enzymejs.github.io/enzyme/) before use.

```@example gaussnewton
using Enzyme
using BenchmarkTools

# 10) Define the log-posterior for logistic regression with a standard normal prior
function obj(β::AbstractVector, X::AbstractMatrix, y::AbstractVector)
    Xβ = X * β
    return mean(y .* Xβ .- log.(1 .+ exp.(Xβ)))
end

# Reverse-mode gradient and forward-over-reverse Hessian via Enzyme
grad_enzyme = (β, X, y) -> Enzyme.gradient(Reverse, obj, β, Const(X), Const(y))[1]
function jacobian_enzyme(β, X, y)
    Enzyme.jacobian(set_runtime_activity(Forward), grad_enzyme, β, Const(X), Const(y))
end

# 11) In-place wrappers expected by Gauss–Newton
function make_logpost!(X, y)
    (out, β) -> (out[1] = obj(β, X, y); out)
end
function make_grad!(X, y)
    function _grad!(out::AbstractVector{T}, β::AbstractVector{T}) where {T}
        out .= grad_enzyme(β, X, y)
        return out
    end
    _grad!
end
function make_hess!(X, y)
    function _hess!(out::AbstractMatrix{T}, β::AbstractVector{T}) where {T}
        J, _ = jacobian_enzyme(β, X, y)
        out .= J
        return out
    end
    _hess!
end

logpostE! = make_logpost!(X, y)
gradE! = make_grad!(X, y)
hessE! = make_hess!(X, y)

inplace_enzyme = ExponentialFamilyProjection.InplaceLogpdfGradHess(logpostE!, gradE!, hessE!)
prj_enzyme = ProjectedTo(MvNormalMeanCovariance, d; parameters = params)
result_enzyme = project_to(prj_enzyme, inplace_enzyme)
```

We can quickly compare the runtime of the Enzyme-based implementation to the manual one defined above.

```@example gaussnewton
# 12) Speed comparison against the manual implementation from above
t_manual = @belapsed project_to($prj, $inplace)
t_enzyme = @belapsed project_to($prj_enzyme, $inplace_enzyme)
speedup = t_manual / t_enzyme
round.((speedup, t_manual, t_enzyme); digits = 3)
```

On typical runs we observe a substantial speedup (often around 10×) for Enzyme while maintaining the same result.

### Closed-form strategy (zero-variance gradients)

The `ClosedFormStrategy` uses `ClosedFormExpectations.jl` to compute exact, analytic gradients without Monte Carlo sampling. This provides "zero-variance" gradients, leading to faster and more accurate convergence compared to sampling-based strategies.

!!! note
    To use `ClosedFormStrategy`, you must install and load `ClosedFormExpectations.jl`:
    ```julia
    using Pkg
    Pkg.add("ClosedFormExpectations")
    using ClosedFormExpectations
    ```

Let's compare `ClosedFormStrategy` with `ControlVariateStrategy` by projecting a LogNormal distribution onto a Gamma distribution:

```@example projection
using ClosedFormExpectations
using BenchmarkTools

# Target: LogNormal(μ=1.0, σ=0.5)
target_dist = LogNormal(1.0, 0.5)
target = (x) -> logpdf(target_dist, x)

# Initial point
initial_dist = Gamma(2.0, 2.0)

# Project using ClosedFormStrategy
t_closed = @elapsed result_closed = project_to(
    ProjectedTo(Gamma; parameters=ProjectionParameters(strategy=ClosedFormStrategy(), niterations=50, tolerance=1e-5)),
    target;
    initialpoint = initial_dist
)

# Project using ControlVariateStrategy  
t_cv = @elapsed result_cv = project_to(
    ProjectedTo(Gamma; parameters=ProjectionParameters(strategy=ControlVariateStrategy(nsamples=500), niterations=50, tolerance=1e-5)),
    target;
    initialpoint = initial_dist
)

println("ClosedFormStrategy time: $(round(t_closed * 1000, digits=2)) ms")
println("ControlVariateStrategy time: $(round(t_cv * 1000, digits=2)) ms")
println("Speedup: $(round(t_cv / t_closed, digits=2))x")
```

Now let's visualize the results to see how both strategies compare:

```@example projection
using Plots

xs = 0.01:0.01:10.0

plot(xs, x -> pdf(target_dist, x), 
     label="Target (LogNormal)", linewidth=2, 
     fill=0, fillalpha=0.2, color=:blue)
plot!(xs, x -> pdf(result_closed, x), 
      label="ClosedForm", linewidth=2, 
      linestyle=:dash, color=:red)
plot!(xs, x -> pdf(result_cv, x), 
      label="ControlVariate", linewidth=2, 
      linestyle=:dot, color=:green)
xlabel!("x")
ylabel!("Density")
title!("LogNormal → Gamma Projection Comparison")
```

The `ClosedFormStrategy` typically provides:
- **Faster convergence**: No Monte Carlo noise in gradients
- **Better accuracy**: Exact gradient computations
- **Speed advantages**: Especially significant for lower-dimensional problems

### Projection with samples

The projection can be done given a set of samples instead of the function directly. For example, let's project an set of samples onto a Beta distribution:

```@example projection
using StableRNGs

hiddenbeta = Beta(10, 3)
samples = rand(StableRNG(42), hiddenbeta, 1_000)
prj = ProjectedTo(Beta)
result = project_to(prj, samples)
@test kldivergence(result, hiddenbeta) < 1e-2 #hide
result #hide
```

```@example projection
plot(0.0:0.01:1.0, x -> pdf(hiddenbeta, x), label="real distribution", fill = 0, fillalpha = 0.2)
histogram!(samples, label = "samples", normalize = :pdf, fillalpha = 0.2)
plot!(0.0:0.01:1.0, x -> pdf(result, x), label="estimated projection", fill = 0, fillalpha = 0.2)
```

## Other 

## Manopt extensions

```@docs 
ExponentialFamilyProjection.ProjectionCostGradientObjective
```

### Bounded direction update rule

The `ExponentialFamilyProjection.jl` package implements a specialized gradient direction rule that limits the norm (manifold-specific) of the gradient to a pre-specified value.

```@docs
ExponentialFamilyProjection.BoundedNormUpdateRule
```

## Index

```@index
```

