```@meta
CurrentModule = ExponentialFamilyProjection
```

# ExponentialFamilyProjection

Documentation for [ExponentialFamilyProjection](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl).

The `ExponentialFamilyProjection.jl` package offers a suite of functions for projecting an arbitrary (un-normalized) log probability density function onto a specified member of the exponential family (e.g., Gaussian, Beta, Bernoulli). This is achieved by optimizing the natural parameters of the exponential family member within a defined manifold. The library leverages `Manopt.jl` for optimization and utilizes [`ExponentialFamilyManifolds.jl`](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl) to define the manifolds corresponding to the members of the exponential family.

## Projection parameters

In order to project a log probability density function onto a member of the exponential family, the user first needs to specify projection parameters:

```@docs
ExponentialFamilyProjection.ProjectionParameters
ExponentialFamilyProjection.DefaultProjectionParameters
```

Read more about different optimization strategies [here](@ref opt-strategies).

## Projection family

After the parameters have been specified the user can proceed with specifying the projection type (exponential family member), its dimensionality and (optionally) the conditioner.

```@docs
ExponentialFamilyProjection.ProjectedTo
```

## Projection 

The projection is performed by calling the `project_to` function with the specified [`ExponentialFamilyProjection.ProjectedTo`](@ref) and log probability density function:

```@docs 
ExponentialFamilyProjection.project_to
```

## [Optimization strategies](@id opt-strategies)

The optimization procedure requires computing the expectation of the gradient to perform gradient descent in the natural parameters space. Currently, the library provides one strategy for computing these expectations:

```@docs
ExponentialFamilyProjection.ControlVariateStrategy
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

## Index

```@index
```

