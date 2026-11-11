# `ExpectationStrategy`: the projection's expectations as a quadrature from a backend of
# BayesBase's expectation protocol, deterministic. Here the backend is a midpoint rule in
# probability space, which serves any continuous univariate distribution.

@testsetup module ExpectationStrategySetup
    using BayesBase, Distributions

    struct Midpoints
        n::Int
    end
    BayesBase.supports_expectation(::Midpoints, ::Type, ::Type{<:ContinuousUnivariateDistribution}) = true
    BayesBase.prepare_expectation(rule::Midpoints, q::ContinuousUnivariateDistribution) =
        BayesBase.ExpectationPointSet(quantile.(q, ((1:rule.n) .- 0.5) ./ rule.n), fill(1 / rule.n, rule.n))

    export Midpoints
end

@testitem "ExpectationStrategy: a conjugate posterior is projected onto itself" setup = [ExpectationStrategySetup] begin
    using BayesBase, Distributions, ExponentialFamily
    import ExponentialFamilyProjection: ExpectationStrategy, ProjectedTo, ProjectionParameters, project_to

    # the midpoint rule's own error is about 1e-3 at this many points
    parameters = ProjectionParameters(strategy = ExpectationStrategy(Midpoints(20000)), niterations = 2000, tolerance = 1.0e-10)

    # Beta(2, 3) times 7 successes in 20 trials is Beta(9, 16)
    logp(p) = logpdf(Beta(2.0, 3.0), p) + 7 * log(p) + 13 * log1p(-p)
    q = project_to(ProjectedTo(Beta; parameters), logp)
    @test q isa Beta
    @test all(isapprox.(params(q), (9.0, 16.0); rtol = 5.0e-3))

    # Gamma(2, rate 1) times Poisson counts summing to 11 over 4 observations is Gamma(13, rate 5)
    logλ(λ) = logpdf(Gamma(2.0, 1.0), λ) + 11 * log(λ) - 4λ
    q = project_to(ProjectedTo(Gamma; parameters), logλ)
    @test mean(q) ≈ 13 / 5 rtol = 5.0e-3
    @test var(q) ≈ 13 / 25 rtol = 1.0e-2
end

@testitem "ExpectationStrategy: no variance, and close to the sampled projection" setup = [ExpectationStrategySetup] begin
    using BayesBase, Distributions, ExponentialFamily
    import ExponentialFamilyProjection: ExpectationStrategy, ControlVariateStrategy, ProjectedTo, ProjectionParameters, project_to

    # a Gaussian target, in the Gaussian family
    logp(x) = logpdf(NormalMeanVariance(1.5, 0.5), x)
    parameters = ProjectionParameters(strategy = ExpectationStrategy(Midpoints(20000)), niterations = 1000, tolerance = 1.0e-10)
    first_run = project_to(ProjectedTo(NormalMeanVariance; parameters), logp)
    second_run = project_to(ProjectedTo(NormalMeanVariance; parameters), logp)
    # the gradient has no variance: only the random starting point differs between runs
    @test mean(first_run) ≈ mean(second_run) rtol = 1.0e-8
    @test var(first_run) ≈ var(second_run) rtol = 1.0e-8
    @test mean(first_run) ≈ 1.5 rtol = 1.0e-3
    @test var(first_run) ≈ 0.5 rtol = 5.0e-3

    sampled = project_to(ProjectedTo(NormalMeanVariance), logp)
    @test mean(sampled) ≈ mean(first_run) atol = 0.1

    @test ExpectationStrategy(Midpoints(10)) == ExpectationStrategy(Midpoints(10))
    @test_throws "callable" project_to(ProjectedTo(NormalMeanVariance; parameters), randn(10))
end
