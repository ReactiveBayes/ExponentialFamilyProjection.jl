@testitem "ProjectedTo structure" begin
    using Distributions, ExponentialFamily, ManifoldsBase, JET

    import ExponentialFamilyProjection:
        get_projected_to_type,
        get_projected_to_dims,
        get_projected_to_conditioner,
        get_projected_to_parameters,
        get_projected_to_manifold

    import ExponentialFamilyProjection:
        getstrategy, getniterations, gettolerance, getstepsize, get_stopping_criterion

    @test repr(ProjectedTo()) ==
          "ProjectedTo(ExponentialFamily.ExponentialFamilyDistribution)"
    @test repr(ProjectedTo(3)) ==
          "ProjectedTo(ExponentialFamily.ExponentialFamilyDistribution, dims = 3)"
    @test repr(ProjectedTo(Beta)) == "ProjectedTo(Distributions.Beta)"
    @test repr(ProjectedTo(MvNormalMeanCovariance, 3)) ==
          "ProjectedTo(ExponentialFamily.MvNormalMeanCovariance, dims = 3)"
    @test repr(ProjectedTo(Laplace, conditioner = 2.0)) ==
          "ProjectedTo(Distributions.Laplace, conditioner = 2.0)"

    @test get_projected_to_type(ProjectedTo()) === ExponentialFamilyDistribution
    @test get_projected_to_dims(ProjectedTo(3)) === (3,)
    @test get_projected_to_conditioner(ProjectedTo()) === nothing

    @test get_projected_to_type(ProjectedTo(Beta)) === Beta
    @test get_projected_to_dims(ProjectedTo(Beta)) === ()
    @test get_projected_to_conditioner(ProjectedTo(Beta)) === nothing
    @test get_projected_to_manifold(ProjectedTo(Beta)) isa AbstractManifold

    @test get_projected_to_type(ProjectedTo(MvNormalMeanCovariance, 3)) ===
          MvNormalMeanCovariance
    @test get_projected_to_dims(ProjectedTo(MvNormalMeanCovariance, 3)) === (3,)
    @test get_projected_to_conditioner(ProjectedTo(MvNormalMeanCovariance, 3)) === nothing
    @test get_projected_to_manifold(ProjectedTo(MvNormalMeanCovariance, 3)) isa
          AbstractManifold

    @test get_projected_to_type(ProjectedTo(Laplace, conditioner = 2.0)) === Laplace
    @test get_projected_to_dims(ProjectedTo(Laplace, conditioner = 2.0)) === ()
    @test get_projected_to_conditioner(ProjectedTo(Laplace, conditioner = 2.0)) === 2.0

    defaultparams = ExponentialFamilyProjection.DefaultProjectionParameters()
    parameters_from_creation = get_projected_to_parameters(ProjectedTo(Beta))

    @test getstrategy(defaultparams) == getstrategy(parameters_from_creation)

    @test getniterations(defaultparams) == getniterations(parameters_from_creation)
    @test gettolerance(defaultparams) == gettolerance(parameters_from_creation)
    # Testing `typeof` here since `Manopt` does not implement `==` 
    @test typeof(getstepsize(defaultparams)) ==
          typeof(getstepsize(parameters_from_creation))
    @test typeof(get_stopping_criterion(defaultparams)) ==
          typeof(get_stopping_criterion(parameters_from_creation))
    # These should pass as soon as `Manopt` implements `==`
    @test_broken getstepsize(defaultparams) == getstepsize(parameters_from_creation)
    @test_broken get_stopping_criterion(defaultparams) ==
                 get_stopping_criterion(parameters_from_creation)

    parameters = ProjectionParameters()
    @test get_projected_to_parameters(ProjectedTo(Beta, parameters = parameters)) ===
          parameters

    @test_opt ProjectedTo()
    @test_opt ProjectedTo(3)
    @test_opt ProjectedTo(MvNormalMeanCovariance, 3)
    @test_opt ProjectedTo(Laplace, conditioner = 2.0)

    @test_throws "The dimensions must be integers, but `Tuple{String}` has been provided. Use `conditioner = ...` keyword argument to supply conditioner." ProjectedTo(
        Beta,
        "hello",
    )
    @test_throws "The dimensions must be integers, but `Tuple{$(typeof(parameters))}` has been provided. Use `conditioner = ...` keyword argument to supply conditioner. Use `parameters = ...` keyword argument to supply parameters." ProjectedTo(
        Beta,
        parameters,
    )
end

@testitem "ProjectionParameters structure" begin
    import ExponentialFamilyProjection:
        ControlVariateStrategy, getniterations, getstrategy, gettolerance

    niterations = rand(Int)
    strategy = ControlVariateStrategy()
    tolerance = rand(Float64)

    parameters = ProjectionParameters(
        niterations = niterations,
        strategy = strategy,
        tolerance = tolerance,
    )

    @test getniterations(parameters) === niterations
    @test getstrategy(parameters) === strategy
    @test gettolerance(parameters) === tolerance
end

@testitem "ProjectionParameters get_stopping_criterion" begin
    using Manopt
    import ExponentialFamilyProjection: ProjectionParameters, get_stopping_criterion

    @testset "ProjectionParameters with iterations but no tolerance" begin
        parameters = ProjectionParameters(niterations = 100, tolerance = missing)
        stopping_criterion = get_stopping_criterion(parameters)
        @test stopping_criterion isa StopAfterIteration
    end

    @testset "ProjectionParameters with tolerance but no iterations" begin
        parameters = ProjectionParameters(niterations = missing, tolerance = 1e-2)
        stopping_criterion = get_stopping_criterion(parameters)
        @test stopping_criterion isa StopWhenGradientNormLess
    end

    @testset "ProjectionParameters with both iterations and tolerance" begin
        parameters = ProjectionParameters(niterations = 100, tolerance = 1e-2)
        stopping_criterion = get_stopping_criterion(parameters)
        @test stopping_criterion isa StopWhenAny{
            <:Tuple{<:Manopt.StopAfterIteration,<:Manopt.StopWhenGradientNormLess},
        }
    end
end

@testitem "Projection should support supplementary natural parameters" begin
    using ExponentialFamily, StableRNGs, BayesBase, Distributions

    rng = StableRNG(42)
    prj = ProjectedTo(
        Beta,
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.ControlVariateStrategy(),
            tolerance = 1e-4,
            niterations = 300,
            rng = rng,
        ),
    )

    for n = 2:15
        distributions = [Beta(1 + 10rand(rng), 1 + 10rand(rng)) for i = 1:n]
        analytical =
            reduce((l, r) -> prod(PreserveTypeProd(Distribution), l, r), distributions)

        targetfn = (x) -> logpdf(distributions[1], x)
        approximated_with_supplementary = project_to(prj, targetfn, distributions[2:end]...)
        approximated_without_supplementary = project_to(prj, targetfn)

        @test kldivergence(approximated_with_supplementary, analytical) < 1e-3
        @test kldivergence(approximated_without_supplementary, analytical) > 0.4
    end
end

@testitem "Projection a product with supplementary natural parameters should better than just `ProductOf`" begin
    using ExponentialFamily, BayesBase, Distributions, JET, StableRNGs
    distributions = [
        (Bernoulli(0.5), Bernoulli(0.5)),
        (Bernoulli(0.1), Bernoulli(0.9)),
        (Bernoulli(0.9), Bernoulli(0.1)),
        (Beta(10, 10), Beta(3, 3)),
        (Beta(1, 1), Beta(0.1, 3)),
        (Normal(0, 1), Normal(0, 1)),
        (NormalMeanVariance(-2, 2), NormalMeanVariance(2, 5)),
        (NormalMeanVariance(3, 20), NormalMeanVariance(0.1, 0.1)),
        (Gamma(1, 1), Gamma(10, 10)),
        # its actually worse for `MvNormalMeanCovariance`
        # (MvNormalMeanCovariance([ 3.14, 2.16 ], [ 1.0 0.0; 0.0 1.0 ]), MvNormalMeanCovariance([ -4.2, 4.2 ], [ 3.14 -0.1; -0.1 4.13 ])),
        (Dirichlet([1, 1]), Dirichlet([2, 2])),
        # its actually worse for `Categorical`
        # (Categorical([0.5, 0.5]), Categorical([0.4, 0.6])),
        # its actually worse for `LogNormal`
        # (LogNormal(-1, 10), LogNormal(3, 4)),
    ]

    for distribution in distributions
        left = distribution[1]
        right = distribution[2]
        dims = size(rand(distribution))

        prj = ProjectedTo(
            ExponentialFamily.exponential_family_typetag(left),
            dims...;
            conditioner = nothing,
            parameters = ProjectionParameters(tolerance = 1e-12, niterations = 2000),
        )

        M = ExponentialFamilyProjection.get_projected_to_manifold(prj)
        initialpoint = rand(StableRNG(42), M)

        targetfn_1 = (x) -> logpdf(left, x)
        targetfn_2 = (x) -> logpdf(ProductOf(left, right), x)
        approximated_1 = project_to(prj, targetfn_1, right, initialpoint = initialpoint)
        approximated_2 = project_to(prj, targetfn_2, initialpoint = initialpoint)
        analytical = prod(PreserveTypeProd(Distribution), left, right)

        @test abs(kldivergence(approximated_1, analytical)) < 1e-2
        @test abs(kldivergence(approximated_2, analytical)) < 1e-2

        @test abs(kldivergence(approximated_1, analytical)) <
              abs(kldivergence(approximated_2, analytical))
    end

    @test_throws "Supplementary distributions must be of the same exponential member as the projection target `Distributions.Beta`, got `Distributions.Bernoulli`" project_to(
        ProjectedTo(Beta),
        (x) -> 1,
        Bernoulli(0.5),
    )
    @test_throws "Supplementary distributions must have the same conditioner as the projection target `Distributions.Laplace` with `conditioner = 2.0`, got `Distributions.Laplace` with `conditioner = 3.0`" project_to(
        ProjectedTo(Laplace, conditioner = 2.0),
        (x) -> 1,
        Laplace(3.0, 0.5),
    )
end

@testitem "Projection should decrease cost" begin

    using ExponentialFamily, BayesBase, Distributions, Manopt, JET

    distributions = [
        (Bernoulli(0.5), Bernoulli(0.5)),
        (Bernoulli(0.1), Bernoulli(0.9)),
        (Bernoulli(0.9), Bernoulli(0.1)),
        (Beta(10, 10), Beta(3, 3)),
        (Beta(1, 1), Beta(0.1, 3)),
        (Normal(0, 1), Normal(0, 1)),
        (NormalMeanVariance(-2, 2), NormalMeanVariance(2, 5)),
        (NormalMeanVariance(3, 20), NormalMeanVariance(0.1, 0.1)),
        (Gamma(1, 1), Gamma(10, 10)),
        (
            MvNormalMeanCovariance([3.14, 2.16], [1.0 0.0; 0.0 1.0]),
            MvNormalMeanCovariance([-4.2, 4.2], [3.14 -0.1; -0.1 4.13]),
        ),
        (Dirichlet([2, 2]), Dirichlet([3, 3])),
        (LogNormal(-1, 10), LogNormal(3, 4)),
        (Chisq(2), Chisq(10)),
    ]

    for distribution in distributions
        left = distribution[1]
        right = distribution[2]
        dims = size(rand(distribution))

        prj = ProjectedTo(
            ExponentialFamily.exponential_family_typetag(left),
            dims...;
            conditioner = nothing,
            parameters = ProjectionParameters(tolerance = 1e-12, niterations = 1000),
        )

        @testset "case 1 with supplementary" begin
            record = [RecordCost()]
            targetfn_1 = (x) -> logpdf(left, x)
            approximated_1 = project_to(prj, targetfn_1, right, record = record)
            recorded_values = record[1].recorded_values
            @test recorded_values[1] > recorded_values[end]
            @test (
                count(v -> v <= 0 || isapprox(v, 0; atol = 1e-6), diff(recorded_values)) /
                (length(recorded_values) - 1)
            ) > 0.7

        end

        @testset "case 2 without supplementary" begin
            record = [RecordCost()]
            targetfn_2 = (x) -> logpdf(ProductOf(left, right), x)
            approximated_2 = project_to(prj, targetfn_2; record = record)
            recorded_values = record[1].recorded_values
            @test recorded_values[1] > recorded_values[end]
            @test (
                count(v -> v <= 0 || isapprox(v, 0; atol = 1e-6), diff(recorded_values)) /
                (length(recorded_values) - 1)
            ) > 0.7
        end
    end
end

@testitem "Test initial point keyword argument" begin
    using ExponentialFamily, ExponentialFamilyManifolds, BayesBase

    dist = Beta(5, 5)
    efdist = convert(ExponentialFamilyDistribution, dist)
    targetfn = (x) -> logpdf(dist, x)

    M = ExponentialFamilyManifolds.get_natural_manifold(Beta, ())
    initialpoint = ExponentialFamilyManifolds.partition_point(
        M,
        getnaturalparameters(convert(ExponentialFamilyDistribution, dist)),
    )

    projection_with_vector =
        project_to(ProjectedTo(Beta), targetfn, initialpoint = initialpoint)
    projection_with_vector_repeated =
        project_to(ProjectedTo(Beta), targetfn, initialpoint = initialpoint)
    projection_with_dist = project_to(ProjectedTo(Beta), targetfn, initialpoint = dist)
    projection_with_dist_repeated =
        project_to(ProjectedTo(Beta), targetfn, initialpoint = dist)
    projection_with_efdist = project_to(ProjectedTo(Beta), targetfn, initialpoint = efdist)
    projection_with_efdist_repeated =
        project_to(ProjectedTo(Beta), targetfn, initialpoint = efdist)

    @test params(projection_with_vector) === params(projection_with_vector_repeated)
    @test params(projection_with_dist) === params(projection_with_dist_repeated)
    @test params(projection_with_vector) === params(projection_with_dist)
    @test params(projection_with_vector) === params(projection_with_efdist)
    @test params(projection_with_vector) === params(projection_with_efdist_repeated)
    @test params(projection_with_dist) === params(projection_with_efdist)

end

@testitem "test_convergence_to_stable_point" begin
    using StableRNGs

    include("projected_to_setuptests.jl")

    rng = StableRNG(42)

    # small variance
    for c in (0, 1, 5, 10), v in (1e-3, 1e-2), n in (20, 100)
        series = map(x -> rand(rng, Normal(1 / x^(0.5) + c, v)), 1:n)
        converged = test_convergence_to_stable_point(series)
        @test converged
    end

    # large variance
    for c in (0, 1, 5, 10), v in (1e+3, 1e+2), n in (20, 100)
        series = map(x -> rand(rng, Normal(1 / x^(0.5) + c, v)), 1:n)
        converged = test_convergence_to_stable_point(series)
        @test !converged
    end
end

@testitem "Projection should be stable against huge gradients" begin
    using StableRNGs, BayesBase, ExponentialFamily, Distributions

    # This example is an adaption of inference in RxInfer.jl package
    # The general idea here is that if we have a variable that is shared 
    # across many nodes in a graph, the resulting posterior is a product of 
    # large number of messages which can lead to numerical instability.

    # This is an analytical message update rule for the Bernoulli factor node
    message_update_rule =
        (observation) -> begin
            analytical =
                Beta(one(observation) + observation, 2one(observation) - observation)
            # However, instead of only returning the analytical result, we will also return a
            # lambda function that will be used to compute the posterior during the projection
            lambda = let analytical = analytical
                (x) -> logpdf(analytical, x)
            end
            # We use the analytical solution only to compare the result in the test
            return analytical, lambda
        end

    rng = StableRNG(42)

    # It should also work for the larger number of messages
    # but then the test takes too much time
    Nmessages_range = (1000, 2000, 5000)

    for Nmessages in Nmessages_range

        prior = Beta(1, 1)
        dist = Bernoulli(0.7)
        data = rand(rng, dist, Nmessages)
        messages = map(message_update_rule, data)
        analytical = map(s -> s[1], messages)
        lambdas = map(s -> s[2], messages)

        analytical_posterior = reduce(
            (l, r) -> prod(PreserveTypeProd(Distribution), l, r),
            analytical;
            init = prior,
        )

        # The idea here is to test the default configuration, which should be able to handle this case 
        # Non-default configuration could already solve this issue by simply reducing the stepsize to a very small value
        projection_config = ProjectedTo(Beta)
        projection_posterior =
            project_to(projection_config, (x) -> logpdf(prior, x) + sum(l -> l(x), lambdas))

        @test all(p -> !isnan(p) && !isinf(p), params(projection_posterior))

        # In the case of the large number of messages the default configuration simply does not 
        # have enough number iterations to converge, the result still should not have `infs` or `nans` though
        if Nmessages < 5_000
            @test mean(projection_posterior) ≈ mean(analytical_posterior) rtol = 1e-1
            @test mode(projection_posterior) ≈ mode(analytical_posterior) rtol = 1e-1
            @test 0 < kldivergence(projection_posterior, analytical_posterior) < 1e-1
        end

    end

end

@testitem "Extreme projections should not produce NaNs" begin
    using BayesBase, ExponentialFamily, Distributions

    @testset "Extreme Beta skewed to right" begin
        extreme = Beta(1e11, 1)
        projection_config = ProjectedTo(Beta)
        projection_posterior = project_to(projection_config, (x) -> logpdf(extreme, x))
        @test !isnan(mean(projection_posterior))
    end

    @testset "Extreme Beta skewed to left" begin
        extreme = Beta(1, 1e11)
        projection_config = ProjectedTo(Beta)
        projection_posterior = project_to(projection_config, (x) -> logpdf(extreme, x))
        @test !isnan(mean(projection_posterior))
    end

    @testset "Normal with extremly small std" begin
        extreme = Normal(0, 1e-11)
        projection_config = ProjectedTo(NormalMeanVariance)
        projection_posterior = project_to(projection_config, (x) -> logpdf(extreme, x))
        @test !isnan(mean(projection_posterior))
    end
end

@testitem "do not produce debug statements by default" begin
    using ExponentialFamilyProjection, StableRNGs, ExponentialFamily, Manopt, JET

    rng = StableRNG(42)
    prj = ProjectedTo(Beta)
    targetfn = (x) -> rand(rng) > 0.5 ? 1 : -1

    @test_logs (:warn, r"The cost increased.*") match_mode = :any project_to(
        prj,
        targetfn,
        debug = [Manopt.DebugWarnIfCostIncreases()],
    )

    # Do not produce debug output by default
    @test_logs match_mode = :all project_to(prj, targetfn)
    @test_logs match_mode = :all project_to(prj, targetfn, debug = [])
    
end

@testitem "Direction rule can improve for MLE" begin 
    using BayesBase, ExponentialFamily, Distributions
    using ExponentialFamilyProjection, StableRNGs

    dists = (Beta(1, 1), Gamma(10, 20), Bernoulli(0.8), NormalMeanVariance(-10, 0.1), Poisson(4.8))
    
    for dist in dists
        rng = StableRNG(42)
        data = rand(rng, dist, 4000)
        
        norm_bounds = [0.01, 0.1, 10.0]
        
        divergences = map(norm_bounds) do norm
            parameters = ProjectionParameters(
                direction = ExponentialFamilyProjection.BoundedNormUpdateRule(norm)
            )
            projection = ProjectedTo(ExponentialFamily.exponential_family_typetag(dist), ()..., parameters = parameters)
            approximated = project_to(projection, data)
            kldivergence(approximated, dist)
        end

        @testset "true dist $(dist)" begin
            @test issorted(divergences, rev=true)
            @test (divergences[1] - divergences[end]) / divergences[1] > 0.05
        end
        
    end
end

@testitem "MomentumGradient direction update rule on logpdf" begin
    using BayesBase, ExponentialFamily, Distributions
    using ExponentialFamilyProjection, ExponentialFamilyManifolds, Manopt, StableRNGs


    true_dist = MvNormal([1.0, 2.0], [1.0 0.7; 0.7 2.0])
    logp = (x) -> logpdf(true_dist, x)

    manifold = ExponentialFamilyManifolds.get_natural_manifold(MvNormalMeanCovariance, (2,), nothing)
    initialpoint = rand(manifold)
    direction = MomentumGradient(manifold, initialpoint)

    momentum_parameters = ProjectionParameters(
        direction = direction,
        niterations = 1000,
        tolerance = 1e-8
    )

    projection = ProjectedTo(MvNormalMeanCovariance, 2, parameters=momentum_parameters)
    
    approximated = project_to(projection, logp, initialpoint = initialpoint)
    
    @test approximated isa MvNormalMeanCovariance
    @test kldivergence(approximated, true_dist) < 0.01
    @test projection.parameters.direction isa MomentumGradient
end

@testitem "MomentumGradient direction update rule on samples" begin
    using BayesBase, ExponentialFamily, Distributions
    using ExponentialFamilyProjection, ExponentialFamilyManifolds, Manopt, StableRNGs

    true_dist = MvNormal([1.0, 2.0], [1.0 0.7; 0.7 2.0])
    rng = StableRNG(42)
    samples = rand(rng, true_dist, 1000)
    
    manifold = ExponentialFamilyManifolds.get_natural_manifold(MvNormalMeanCovariance, (2,), nothing)
    
    initialpoint = rand(rng, manifold)
    direction = MomentumGradient(manifold, initialpoint)
    
    momentum_parameters = ProjectionParameters(
        direction = direction,
        niterations = 1000,
        tolerance = 1e-8
    )
    
    projection = ProjectedTo(MvNormalMeanCovariance, 2, parameters=momentum_parameters)
    approximated = project_to(projection, samples, initialpoint = initialpoint)
    
    @test approximated isa MvNormalMeanCovariance
    @test kldivergence(approximated, true_dist) < 0.01  # Ensure good approximation
    @test projection.parameters.direction isa MomentumGradient
end
