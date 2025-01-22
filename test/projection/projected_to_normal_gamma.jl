@testitem "Simple projection to `NormalGamma`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = NormalGamma(1.0, 1.0, 1.0, 1.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = NormalGamma(5.0, 1.0, 2.0, 1.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = NormalGamma(5.0, 3.0, 2.0, 1.0)
        @test test_projection_convergence(distribution)
    end

end

@testitem "MLE" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection, LinearAlgebra

    include("./projected_to_setuptests.jl")

    @testset let distribution = NormalGamma(1.0, 1.0, 1.0, 1.0)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = NormalGamma(5.0, 1.0, 2.0, 1.0)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = NormalGamma(5.0, 3.0, 2.0, 1.0)
        @test test_projection_mle(distribution)
    end

end