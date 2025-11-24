export ClosedFormStrategy

"""
    ClosedFormStrategy <: ExponentialFamilyProjection.AbstractStrategy

A projection strategy that uses `ClosedFormExpectations.jl` to compute the exact gradient 
of the cross-entropy term \$\\mathbb{E}_{q_\\eta}[\\log \\tilde{p}(x)]\$ analytically.

This strategy provides a "Zero-Variance" gradient estimator, avoiding the noise associated 
with Monte Carlo sampling (like in `ControlVariateStrategy`).

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

# Example

```julia
using ExponentialFamilyProjection, ClosedFormExpectations
using Distributions

# Target distribution
target = LogNormal(1.0, 0.5)

# Project to Gamma using closed-form gradients
result = project_to(
    ProjectedTo(
        Gamma;
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            niterations = 50
        )
    ),
    target
)
```

# References

This estimator was proposed in [Lukashchuk et al., 2024](https://proceedings.mlr.press/v246/lukashchuk24a.html).

!!! note
    This strategy requires that `ClosedFormExpectations.jl` implements `ClosedWilliamsProduct` 
    for the specific pair of target distribution and variational family you're using.
    See the `ClosedFormExpectations.jl` documentation for supported combinations.
"""
struct ClosedFormStrategy end
