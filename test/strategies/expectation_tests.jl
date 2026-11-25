# `ExpectationStrategy`: the projection's expectations as a quadrature from a backend of
# BayesBase's expectation protocol, deterministic. Here the backend is a midpoint rule in
# probability space, which serves any continuous univariate distribution.

@testsetup module ExpectationStrategySetup
using BayesBase, Distributions

struct Midpoints
    n::Int
end
BayesBase.supports_expectation(
    ::Midpoints,
    ::Type,
    ::Type{<:ContinuousUnivariateDistribution},
) = true
BayesBase.prepare_expectation(rule::Midpoints, q::ContinuousUnivariateDistribution) =
    BayesBase.ExpectationPointSet(
        quantile.(q, ((1:rule.n) .- 0.5) ./ rule.n),
        fill(1 / rule.n, rule.n),
    )

export Midpoints
end

@testitem "ExpectationStrategy: a conjugate posterior is projected onto itself" setup =
    [ExpectationStrategySetup] begin
    using BayesBase, Distributions, ExponentialFamily
    import ExponentialFamilyProjection:
        ExpectationStrategy, ProjectedTo, ProjectionParameters, project_to

    # the midpoint rule's own error is about 1e-3 at this many points
    parameters = ProjectionParameters(
        strategy = ExpectationStrategy(Midpoints(20000)),
        niterations = 2000,
        tolerance = 1.0e-10,
    )

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

@testitem "ExpectationStrategy: no variance, and close to the sampled projection" setup =
    [ExpectationStrategySetup] begin
    using BayesBase, Distributions, ExponentialFamily
    import ExponentialFamilyProjection:
        ExpectationStrategy,
        ControlVariateStrategy,
        ProjectedTo,
        ProjectionParameters,
        project_to

    # a Gaussian target, in the Gaussian family
    logp(x) = logpdf(NormalMeanVariance(1.5, 0.5), x)
    parameters = ProjectionParameters(
        strategy = ExpectationStrategy(Midpoints(20000)),
        niterations = 1000,
        tolerance = 1.0e-10,
    )
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
    @test_throws "callable" project_to(
        ProjectedTo(NormalMeanVariance; parameters),
        randn(10),
    )
end

@testitem "ExpectationStrategy: a constant added to the target changes nothing, at small shapes too" begin
    using BayesBase, Distributions, ExponentialFamily
    import ExponentialFamilyProjection:
        ExpectationStrategy, ProjectedTo, ProjectionParameters, project_to
    import ExpectationApproximations: EADefault

    # The step is the regression of the target's log-density on the sufficient statistics over
    # the backend's points: an added constant drops out, and a target in the family is reached
    # exactly, even where the quadrature is not exact for the sufficient statistics (log p, log x)
    parameters = ProjectionParameters(
        strategy = ExpectationStrategy(EADefault()),
        niterations = 500,
        tolerance = 1.0e-12,
    )
    for C in (-100.0, 0.0, 100.0)
        # Beta(0.5, 0.5) times one success is Beta(1.5, 0.5)
        beta = project_to(
            ProjectedTo(Beta; parameters),
            p -> logpdf(Beta(0.5, 0.5), p) + log(p) + C,
        )
        @test all(isapprox.(params(beta), (1.5, 0.5); rtol = 1.0e-6))
        # Gamma(0.7, rate 1) times a Poisson count of 0 is Gamma(0.7, rate 2)
        gamma = project_to(
            ProjectedTo(Gamma; parameters),
            λ -> logpdf(Gamma(0.7, 1.0), λ) - λ + C,
        )
        @test shape(gamma) ≈ 0.7 rtol = 1.0e-6
        @test rate(gamma) ≈ 2.0 rtol = 1.0e-6
    end

    # With Gauss–Jacobi and Gauss–Laguerre rules, which are not exact for log p or log x at these
    # shapes, the step is still exact for a target in the family. The score form of the gradient
    # gave Beta(1.74, 0.63) and Gamma(0.87, rate 2.44) here, and moved with the constant.
    import ExpectationApproximations: GaussJacobi, GaussLaguerre
    polynomial = ProjectionParameters(
        strategy = ExpectationStrategy(
            BayesBase.ExpectationFirstOf(GaussJacobi(16), GaussLaguerre(16)),
        ),
        niterations = 500,
        tolerance = 1.0e-12,
    )
    for C in (-100.0, 0.0, 100.0)
        beta = project_to(
            ProjectedTo(Beta; parameters = polynomial),
            p -> logpdf(Beta(0.5, 0.5), p) + log(p) + C,
        )
        @test all(isapprox.(params(beta), (1.5, 0.5); rtol = 1.0e-6))
        gamma = project_to(
            ProjectedTo(Gamma; parameters = polynomial),
            λ -> logpdf(Gamma(0.7, 1.0), λ) - λ + C,
        )
        @test shape(gamma) ≈ 0.7 rtol = 1.0e-6
        @test rate(gamma) ≈ 2.0 rtol = 1.0e-6
    end

    # A distribution is not a log-density
    @test_throws "Base.Fix1(logpdf" project_to(
        ProjectedTo(Gamma; parameters),
        Gamma(2.0, 1.0),
    )
end

@testitem "ExpectationStrategy: a non-conjugate Gamma target, against the minimiser of the KL divergence" begin
    using BayesBase, Distributions, ExponentialFamily, ForwardDiff, LinearAlgebra
    import ExponentialFamilyProjection:
        ExpectationStrategy, ProjectedTo, ProjectionParameters, project_to
    import ExpectationApproximations: EADefault

    # p̃(λ) = Gamma(3, 1)(λ) / (1 + λ²). The reference minimises KL(q ‖ p̃) over Gamma(a, θ) by
    # Newton's method in (log a, log θ), with the expectation of log p̃ by the trapezoidal rule on
    # a fine grid in u = log λ
    logp(λ) = logpdf(Gamma(3.0, 1.0), λ) - log1p(λ^2)
    us = range(-30.0, 6.0; length = 40_001)
    function kl(v)
        a, θ = exp(v[1]), exp(v[2])
        q = Gamma(a, θ)
        cross = step(us) * sum(u -> pdf(q, exp(u)) * exp(u) * logp(exp(u)), us)
        return -entropy(q) - cross
    end
    v = [log(2.0), log(0.5)]
    for _ = 1:30
        g = ForwardDiff.gradient(kl, v)
        v = v - ForwardDiff.hessian(kl, v) \ g
    end
    reference = Gamma(exp(v[1]), exp(v[2]))

    parameters = ProjectionParameters(
        strategy = ExpectationStrategy(EADefault()),
        niterations = 500,
        tolerance = 1.0e-12,
    )
    q = project_to(ProjectedTo(Gamma; parameters), logp)
    @test mean(q) ≈ mean(reference) rtol = 5.0e-3
    @test var(q) ≈ var(reference) rtol = 5.0e-3
end
