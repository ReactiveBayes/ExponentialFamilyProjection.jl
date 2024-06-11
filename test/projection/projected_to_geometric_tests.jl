@testitem "Simple projection to `Geometric`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Geometric(0.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Geometric(0.25)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Geometric(0.5)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Geometric(0.75)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Geometric(0.9)
        @test test_projection_convergence(distribution)
    end
end