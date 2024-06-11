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

    @testset let distribution = Normal(0.5, 1)
        @test test_projection_convergence(distribution, to = Beta)
    end
end