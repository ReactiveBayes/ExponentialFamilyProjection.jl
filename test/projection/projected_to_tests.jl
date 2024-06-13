@testitem "ProjectedTo structure" begin
    using Distributions, ExponentialFamily, ManifoldsBase, JET

    import ExponentialFamilyProjection:
        get_projected_to_type,
        get_projected_to_dims,
        get_projected_to_conditioner,
        get_projected_to_parameters,
        get_projected_to_manifold

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

    @test get_projected_to_parameters(ProjectedTo(Beta)) ===
          ExponentialFamilyProjection.DefaultProjectionParameters
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
    import ExponentialFamilyProjection: getniterations, getnsamples, gettolerance, getseed

    niterations = rand(Int)
    nsamples = rand(Int)
    tolerance = rand(Float64)
    seed = rand(UInt)

    parameters = ProjectionParameters(
        niterations = niterations,
        nsamples = nsamples,
        tolerance = tolerance,
        seed = seed,
    )

    @test getniterations(parameters) === niterations
    @test getnsamples(parameters) === nsamples
    @test gettolerance(parameters) === tolerance
    @test getseed(parameters) === seed
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
        parameters = ProjectionParameters(tolerance = 1e-4, niterations = 200),
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

@testitem "test_convergence_to_stable_point" begin
    using StableRNGs

    include("projected_to_setuptests.jl")

    rng = StableRNG(42)
    for c in (0, 1, 5, 10), v in (1e-3, 1e-2), n in (20, 100)
        series = map(x -> rand(Normal(1 / x^(0.5) + c, v)), 1:n)
        @test test_convergence_to_stable_point(series)
    end
end