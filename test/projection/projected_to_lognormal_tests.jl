@testitem "Simple projection to `LogNormal`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = LogNormal()
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = LogNormal(0.1, 2.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = LogNormal(-3.14, 2.71)
        @test test_projection_convergence(distribution)
    end
end