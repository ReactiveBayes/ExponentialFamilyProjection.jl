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
    @test typeof(getstepsize(defaultparams)) == typeof(getstepsize(parameters_from_creation))
    @test typeof(get_stopping_criterion(defaultparams)) == typeof(get_stopping_criterion(parameters_from_creation))
    # These should pass as soon as `Manopt` implements `==`
    @test_broken getstepsize(defaultparams) == getstepsize(parameters_from_creation)
    @test_broken get_stopping_criterion(defaultparams) == get_stopping_criterion(parameters_from_creation)

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

@testitem "ProjectionParameters usebuffer" begin
    using Bumper
    import ExponentialFamilyProjection: getstepsize, with_buffer

    parameters = ProjectionParameters(usebuffer = Val(true))

    result = with_buffer(parameters) do buffer
        @test buffer !== nothing
        @no_escape buffer begin
            container = @alloc(Float64, 10)
            @test length(container) === 10
        end
        return "asd"
    end
    @test result == "asd"

    parameters = ProjectionParameters(usebuffer = Val(false))

    result = with_buffer(parameters) do buffer
        @test buffer === nothing
        return "dsa"
    end
    @test result == "dsa"

end

@testitem "Projection result should not depend on the usage of buffer" begin
    using ExponentialFamily, BayesBase
    distributions = [
        Beta(10, 10),
        Gamma(10, 10),
        Exponential(1),
        LogNormal(0, 1),
        Dirichlet([1, 1]),
        NormalMeanVariance(0.0, 1.0),
        MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
    ]

    for distribution in distributions
        parameters_with_buffer = ProjectionParameters(usebuffer = Val(true))
        parameters_without_buffer = ProjectionParameters(usebuffer = Val(false))

        dims = size(rand(distribution))

        prj_with_buffer = ProjectedTo(
            ExponentialFamily.exponential_family_typetag(distribution),
            dims...;
            parameters = parameters_with_buffer,
        )
        prj_without_buffer = ProjectedTo(
            ExponentialFamily.exponential_family_typetag(distribution),
            dims...;
            parameters = parameters_without_buffer,
        )

        targetfn = (x) -> logpdf(distribution, x)
        result_with_buffer = project_to(prj_with_buffer, targetfn)
        result_without_buffer = project_to(prj_without_buffer, targetfn)

        # Small differences are allowed due to different LinearAlgebra routines
        @test result_with_buffer ≈ result_without_buffer
    end
end

@testitem "Projection should support supplementary natural parameters" begin
    using ExponentialFamily, StableRNGs, BayesBase, Distributions

    rng = StableRNG(42)
    prj = ProjectedTo(
        Beta,
        parameters = ProjectionParameters(
            tolerance = 1e-4,
            niterations = 300,
            strategy = ExponentialFamilyProjection.ControlVariateStrategy(rng = rng),
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
    using ExponentialFamily, BayesBase, Distributions, JET
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
            parameters = ProjectionParameters(tolerance = 1e-12, niterations = 1000),
        )

        targetfn_1 = (x) -> logpdf(left, x)
        targetfn_2 = (x) -> logpdf(ProductOf(left, right), x)
        approximated_1 = project_to(prj, targetfn_1, right)
        approximated_2 = project_to(prj, targetfn_2)
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
    for c in (0, 1, 5, 10), v in (1e-3, 1e-2), n in (20, 100)
        series = map(x -> rand(rng, Normal(1 / x^(0.5) + c, v)), 1:n)
        converged, _ = test_convergence_to_stable_point(series)
        @test converged
    end
end

@testitem "test_convergence_nsamples" begin
    include("projected_to_setuptests.jl")
    
    @testset "test_convergence_nsamples tracks divergence" begin
        distribution = Normal(0, 1)
        nsamples_range = 
        diverged_series = map(1:100) do nsamples
            approximation = Normal(nsamples, nsamples)
            (kldivergence(approximation, distribution), approximation)
        end 
        test_result, series = test_convergence_nsamples(distribution, (x) -> 1, NormalMeanVariance, (), missing, experiment = diverged_series)
        @test !test_result
    end
    
    @testset "test_convergence_nsamples tracks cpnvergence" begin
        distribution = Normal(0, 1)
        nsamples_range = 
        convergent_series = map(1:100) do nsamples
            approximation = Normal(0, 1 + 1/nsamples)
            (kldivergence(approximation, distribution), approximation)
        end 
        test_result, series = test_convergence_nsamples(distribution, (x) -> 1, NormalMeanVariance, (), missing, experiment = convergent_series)
        @test test_result
    end     
end

@testitem "test_convergence_niterations" begin
    include("projected_to_setuptests.jl")
    
    @testset "test_convergence_niterations tracks divergence" begin
        distribution = Normal(0, 1)
        nsamples_range = 
        diverged_series = map(1:100) do niterations
            approximation = Normal(niterations, niterations)
            (kldivergence(approximation, distribution), approximation)
        end 
        test_result, series = test_convergence_niterations(distribution, (x) -> 1, NormalMeanVariance, (), missing, experiment = diverged_series)
        @test !test_result
    end
    
    @testset "test_convergence_niterations tracks convergence" begin
        distribution = Normal(0, 1)
        nsamples_range = 
        convergent_series = map(1:100) do niterations
            approximation = Normal(0, 1 + 1/niterations)
            (kldivergence(approximation, distribution), approximation)
        end 
        test_result, series = test_convergence_niterations(distribution, (x) -> 1, NormalMeanVariance, (), missing, experiment = convergent_series)
        @test test_result
    end    
end