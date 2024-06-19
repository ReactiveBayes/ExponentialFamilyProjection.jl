@testitem "Simple projection to `Beta`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Beta(1, 1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Beta(1, 10)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Beta(10, 1)
        @test test_projection_convergence(distribution)
    end
end

@testitem "Project `Normal` to `Beta`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Normal(0.5, 0.1)
        @test test_projection_convergence(distribution, to = Beta)
    end
end

@testitem "Project a product of `Beta` and `Beta` to `Beta`" begin
    using BayesBase, ExponentialFamily, Distributions

    include("./projected_to_setuptests.jl")

    @testset let distribution = ProductOf(Beta(1, 1), Beta(3, 3))
        @test test_projection_convergence(
            distribution,
            to = Beta,
            dims = (),
            conditioner = nothing,
        )
    end

    @testset let distribution = ProductOf(Beta(5, 1), Beta(6, 2))
        @test test_projection_convergence(
            distribution,
            to = Beta,
            dims = (),
            conditioner = nothing,
        )
    end

end

@testitem "Project a product of `Normal` and `Normal` to `Beta`" begin
    using BayesBase, ExponentialFamily, Distributions

    include("./projected_to_setuptests.jl")

    @testset let distribution = ProductOf(Normal(0.2, 0.1), Normal(0.3, 0.1))
        @test test_projection_convergence(
            distribution,
            to = Beta,
            dims = (),
            conditioner = nothing,
        )
    end

    @testset let distribution = ProductOf(Normal(0.5, 1), Normal(0.5, 0.2))
        @test test_projection_convergence(
            distribution,
            to = Beta,
            dims = (),
            conditioner = nothing,
        )
    end

end

