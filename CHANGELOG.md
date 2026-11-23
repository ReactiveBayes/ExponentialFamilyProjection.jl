# Changelog

All notable changes to ExponentialFamilyProjection.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Changes before 3.5.0 are listed in
the [GitHub releases](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl/releases).

## [Unreleased]

The changes below are to be released as 3.5.0.

### Added

- `ExpectationStrategy(backend)`, a deterministic projection strategy. It computes the
  expectations of the projection with `backend`, a backend of BayesBase's expectation protocol
  such as ExpectationApproximations' `EADefault()`, instead of sampling. At every iteration the
  backend gives a point set for the current member of the family. The gradient is the score form
  `Σᵢ wᵢ (log p̃(xᵢ) - log h(xᵢ)) (T(xᵢ) - ∇A(η))`, the step uses the inverse Fisher
  information, and the cost is the negative evidence lower bound. The gradient has no variance,
  so the iterations can stop at a tight tolerance. On a Beta–Bernoulli target whose exact
  posterior is `Beta(9, 16)`, the strategy returns that Beta, and the default sampled strategy
  gives about `Beta(8.0, 14.3)`. The argument of `project_to` must be callable.

### Changed

- **Breaking for the `ClosedFormStrategy` extension:** it requires `ClosedFormExpectations` 1.0,
  which computes through BayesBase's expectation protocol. The extension computes the cost with
  `BayesBase.expectation(ClosedFormExpectation(), f, q)` and the gradient with
  `BayesBase.expectation_gradient(ClosedFormExpectation(backend), f, q)`. A target can be a
  distribution, its log-density `Base.Fix1(logpdf, p)` or a closure capturing a distribution;
  ClosedFormExpectations' `Logpdf` wrapper no longer exists. With ClosedFormExpectations 0.4,
  the extension of 3.4 applies.
- Requires BayesBase 1.7, which defines the expectation protocol.

[Unreleased]: https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl/compare/v3.4.0...HEAD
