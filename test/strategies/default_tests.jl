@testitem "`ProjectedTo` should automatically decide on the best strategy based on the projection argument" begin
    using ExponentialFamily, StableRNGs, BayesBase

    distribution = Beta(1, 1)

    projection_argument_1 = (x) -> logpdf(distribution, x)
    projection_argument_2 = rand(StableRNG(42), distribution, 1_000)

    @testset "Without explicitly specifying the default strategy" begin
        prj = ProjectedTo(Beta)

        result_1 = project_to(prj, projection_argument_1)
        result_2 = project_to(prj, projection_argument_2)

        @test result_1 ≈ distribution atol = 1e-1
        @test result_2 ≈ distribution atol = 1e-1
    end

    @testset "With explicitly specifying the default strategy" begin
        prj = ProjectedTo(
            Beta;
            parameters = ProjectionParameters(
                strategy = ExponentialFamilyProjection.DefaultStrategy(),
            ),
        )

        result_1 = project_to(prj, projection_argument_1)
        result_2 = project_to(prj, projection_argument_2)

        @test result_1 ≈ distribution atol = 1e-1
        @test result_2 ≈ distribution atol = 1e-1
    end
end