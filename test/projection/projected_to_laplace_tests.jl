@testitem "Simple projection to `Laplace`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Laplace()
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Laplace(0.1, 2.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Laplace(-3.14, 2.71)
        @test test_projection_convergence(distribution)
    end
end