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
    import ExponentialFamilyProjection: getniterations, getnsamples

    niterations = rand(Int)
    nsamples = rand(Int)

    parameters = ProjectionParameters(niterations = niterations, nsamples = nsamples)

    @test getniterations(parameters) === niterations
    @test getnsamples(parameters) === nsamples
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