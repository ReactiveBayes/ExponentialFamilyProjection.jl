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
  backend gives a point set for the current member of the family. The step is towards the
  weighted regression `Cov_w(T, T)⁻¹ Cov_w(T, log p̃ - log h)` of the target's log-density on the
  sufficient statistics, both covariances over the backend's points, so it does not change when a
  constant is added to `log p̃`, and a target in the family is projected onto itself whatever the
  quadrature's accuracy on `T`. The cost is the negative evidence lower bound. The step has no
  variance; its accuracy is that of the quadrature on the target, which for Gamma and Beta
  families at shapes below 1 needs a rule in `log x` or `logit x`, as ExpectationApproximations'
  default uses. On a Beta–Bernoulli target whose exact posterior is `Beta(9, 16)`, the strategy
  returns that Beta, and the default sampled strategy gives about `Beta(8.0, 14.3)`. The argument
  of `project_to` must be callable; a distribution is rejected with an error that names
  `Base.Fix1(logpdf, distribution)`.

### Changed

- **Breaking for the `ClosedFormStrategy` extension:** it requires `ClosedFormExpectations` 1.0,
  which computes through BayesBase's expectation protocol. The extension computes the cost with
  `BayesBase.expectation(ClosedFormExpectation(), f, q)` and the gradient with
  `BayesBase.expectation_gradient(ClosedFormExpectation(), f, q)`.
  `ClosedFormStrategy(EnzymeBackend())` becomes `ClosedFormStrategy(AutoEnzyme())`, with
  `AutoEnzyme` from ADTypes.jl: the gradient is then
  `ExpectationFirstOf(ClosedFormExpectation(), ClosedFormAD(AutoEnzyme()))`, the hand-written
  gradient where there is one and the derivative of the closed-form expectation otherwise.
  `AutoEnzyme(mode = Enzyme.Forward)` replaces `EnzymeBackend(EnzymeForward())`. A target can be a
  distribution, its log-density `Base.Fix1(logpdf, p)` or a closure capturing a distribution;
  ClosedFormExpectations' `Logpdf` wrapper no longer exists. With ClosedFormExpectations 0.4,
  the extension of 3.4 applies.
- Requires BayesBase 1.7, which defines the expectation protocol.

[Unreleased]: https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl/compare/v3.4.0...HEAD
