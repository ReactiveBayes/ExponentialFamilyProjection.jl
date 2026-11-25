export ClosedFormStrategy

"""
    ClosedFormStrategy{B} <: ExponentialFamilyProjection.AbstractStrategy

A projection strategy that computes the gradient of the cross-entropy term
\$\\mathbb{E}_{q_\\eta}[\\log \\tilde{p}(x)]\$ in closed form, with `ClosedFormExpectations.jl`.
The gradient has no Monte Carlo variance, unlike the gradient of `ControlVariateStrategy`.

The gradient is `BayesBase.expectation_gradient(backend, f, q)`, the gradient of the expectation
with respect to the natural parameters of `q`. With `backend = nothing`, the default, it uses the
gradients that `ClosedFormExpectations.jl` implements by hand, `ClosedFormExpectation()`. With an
automatic differentiation backend from ADTypes.jl, such as `AutoEnzyme()` with Enzyme.jl loaded,
it uses `BayesBase.ExpectationFirstOf(ClosedFormExpectation(), ClosedFormAD(backend))`: the
hand-written gradient where there is one, and otherwise the derivative of the closed-form
expectation, so the strategy also works for pairs whose expectation has a closed form but whose
gradient is not implemented.

# Requirements

The strategy needs `ClosedFormExpectations.jl` 1.0 or later:

```julia
using ClosedFormExpectations
```

Loading `ClosedFormExpectations` triggers a package extension that implements the gradient
computation for this strategy.

The target is a distribution, its log-density `Base.Fix1(logpdf, p)`, or a closure that captures a
distribution or a `ProductOf`, as RxInfer passes it.

For `Categorical`, expectations are exact sums over all categories, and EFP applies the analytic
inverse Fisher in the first `K-1` coordinates, keeping the reference coordinate fixed at zero.
Callable log-scores, including plain functions and closures, are supported in addition to
distribution targets. This costs O(K) target evaluations per expectation; `ControlVariateStrategy`
remains the default sampled alternative.

# When to Use

Use `ClosedFormStrategy` when:
- You need exact, deterministic gradients without Monte Carlo variance
- The target-to-variational family pair is supported by `ClosedFormExpectations.jl`
- You want faster convergence with fewer iterations
- Reproducibility is critical (no random sampling)

Use `ClosedFormStrategy(AutoEnzyme())` when the expectation has a closed form for the pair but
its gradient is not implemented. Enzyme then differentiates the expectation, using
\$\\nabla_\\eta \\mathbb{E}_q[f] = \\mathbb{E}_q[f \\nabla_\\eta \\log q]\$.

# Examples

```julia
using ExponentialFamilyProjection, ClosedFormExpectations
using Distributions

# Target distribution
target = LogNormal(1.0, 0.5)

# Project to Gamma with the hand-written closed-form gradients
result = project_to(
    ProjectedTo(
        Gamma;
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            niterations = 50
        )
    ),
    Base.Fix1(logpdf, target)
)
```

```julia
using ExponentialFamilyProjection, ClosedFormExpectations, Enzyme
using ADTypes: AutoEnzyme
using Distributions

# Gamma → LogNormal: the expectation has a closed form but its gradient is not implemented,
# so Enzyme differentiates the expectation
target = Gamma(2.0, 1.0)

result = project_to(
    ProjectedTo(
        LogNormal;
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(AutoEnzyme()),
            niterations = 50
        )
    ),
    Base.Fix1(logpdf, target)
)
```

# References

This estimator was proposed in [Lukashchuk et al., 2024](https://proceedings.mlr.press/v246/lukashchuk24a.html).

!!! note
    Without a backend, the strategy needs a closed-form gradient for the pair,
    `BayesBase.supports_expectation_gradient(ClosedFormExpectation(), f, q)`. With
    `AutoEnzyme()`, a closed-form expectation under a univariate exponential family member is
    enough. See the `ClosedFormExpectations.jl`
    documentation for the supported pairs.
"""
struct ClosedFormStrategy{B}
    backend::B
end
ClosedFormStrategy() = ClosedFormStrategy(nothing)
