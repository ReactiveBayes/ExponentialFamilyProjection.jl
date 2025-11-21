export ClosedFormStrategy

"""
    ClosedFormStrategy <: ExponentialFamilyProjection.AbstractStrategy

A projection strategy that uses `ClosedFormExpectations.jl` to compute the exact gradient 
of the cross-entropy term \$\\mathbb{E}_{q_\\eta}[\\log \\tilde{p}(x)]\$ analytically.

This strategy provides a "Zero-Variance" gradient estimator, avoiding the noise associated 
with Monte Carlo sampling (like in `ControlVariateStrategy`).

This estimator was proposed in [Lukashchuk et al., 2024](https://proceedings.mlr.press/v246/lukashchuk24a.html).

!!! note
    To use this strategy, you **must** load the `ClosedFormExpectations` package in your environment.
    Loading `ClosedFormExpectations` will trigger the package extension that implements `compute_gradient!` 
    and `compute_cost` for this strategy.

    It requires that `ClosedFormExpectations.jl` implements `ClosedWilliamsProduct` for the 
    specific pair of target function and variational family.
"""
struct ClosedFormStrategy end
