export ClosedFormStrategy

"""
    ClosedFormStrategy{B} <: ExponentialFamilyProjection.AbstractStrategy

A projection strategy that uses `ClosedFormExpectations.jl` to compute the exact gradient
of the cross-entropy term \$\\mathbb{E}_{q_\\eta}[\\log \\tilde{p}(x)]\$ analytically.

This strategy provides a "Zero-Variance" gradient estimator, avoiding the noise associated
with Monte Carlo sampling (like in `ControlVariateStrategy`).

The optional `backend` field selects the differentiation backend used for computing
`ClosedWilliamsProduct`. When `backend = nothing` (the default), hand-coded closed-form
implementations are used. When an `EnzymeBackend` is provided, Enzyme.jl automatically
differentiates the `ClosedFormExpectation` to obtain the Williams product gradient, enabling
the strategy to work for any target-variational pair where the expectation is implemented
but the Williams product is not.

# Requirements

To use this strategy, you **must** load the `ClosedFormExpectations` package:

```julia
using ClosedFormExpectations
```

Loading `ClosedFormExpectations` will trigger a package extension that implements
the gradient computation for this strategy.

# When to Use

Use `ClosedFormStrategy` when:
- You need exact, deterministic gradients without Monte Carlo variance
- The target-to-variational family pair is supported by `ClosedFormExpectations.jl`
- You want faster convergence with fewer iterations
- Reproducibility is critical (no random sampling)

Use `ClosedFormStrategy(EnzymeBackend())` when:
- A `ClosedFormExpectation` is implemented for the pair, but `ClosedWilliamsProduct` is not
- You want to exploit the identity \$\\nabla_\\eta \\mathbb{E}_q[f] = \\mathbb{E}_q[f \\nabla_\\eta \\log q]\$ via autodiff

# Examples

```julia
using ExponentialFamilyProjection, ClosedFormExpectations
using Distributions

# Target distribution
target = LogNormal(1.0, 0.5)

# Project to Gamma using closed-form gradients (hand-coded Williams product)
result = project_to(
    ProjectedTo(
        Gamma;
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            niterations = 50
        )
    ),
    Logpdf(target)
)
```

```julia
using ExponentialFamilyProjection, ClosedFormExpectations, Enzyme
using Distributions

# Target distribution (Gamma → LogNormal: ClosedFormExpectation is available
# but ClosedWilliamsProduct is not, so we use EnzymeBackend to autodiff it)
target = Gamma(2.0, 1.0)

result = project_to(
    ProjectedTo(
        LogNormal;
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(EnzymeBackend()),
            niterations = 50
        )
    ),
    Logpdf(target)
)
```

# References

This estimator was proposed in [Lukashchuk et al., 2024](https://proceedings.mlr.press/v246/lukashchuk24a.html).

!!! note
    Without a backend, this strategy requires that `ClosedFormExpectations.jl` implements
    `ClosedWilliamsProduct` for the specific target-variational pair. With an `EnzymeBackend`,
    it suffices to have `ClosedFormExpectation` implemented. See the `ClosedFormExpectations.jl`
    documentation for supported combinations.
"""
struct ClosedFormStrategy{B}
    backend::B
end
ClosedFormStrategy() = ClosedFormStrategy(nothing)
