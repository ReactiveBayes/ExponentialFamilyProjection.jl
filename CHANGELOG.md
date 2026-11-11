# Changelog

All notable changes to ExponentialFamilyProjection.jl are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Changes before 3.5.0 are listed in
the [GitHub releases](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl/releases).

## [Unreleased]

The changes below are to be released as 3.5.0.

### Added

- `ExpectationStrategy(backend)`, a deterministic projection strategy. The projection's
  expectations are a quadrature from `backend`, any backend of BayesBase's expectation protocol,
  such as ExpectationApproximations' `EADefault()`, instead of samples. At every iteration the
  backend gives a point set for the current member of the family; the gradient is the score form
  `Σᵢ wᵢ (log p̃(xᵢ) - log h(xᵢ)) (T(xᵢ) - ∇A(η))`, the step goes through the inverse Fisher
  information, and the cost is the negative evidence lower bound. Without random numbers the
  gradient has no variance, so the iterations can stop at a tight tolerance: on a Beta–Bernoulli
  target whose exact posterior is `Beta(9, 16)`, it returns that Beta, where the default sampled
  strategy gives about `Beta(8.0, 14.3)`. The argument of `project_to` must be callable.

### Changed

- BayesBase 1.7 is required, for its expectation protocol.

[Unreleased]: https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl/compare/v3.4.0...HEAD
