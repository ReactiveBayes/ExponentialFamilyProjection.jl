@testitem "Simple projection to `Bernoulli`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Bernoulli(0.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.25)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.5)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.75)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.9)
        @test test_projection_convergence(distribution)
    end
end