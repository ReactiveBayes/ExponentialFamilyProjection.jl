@testitem "Simple projection to `InverseGamma`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = InverseGamma(1, 0.05)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = InverseGamma(1, 0.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = InverseGamma(1, 0.5)
        @test test_projection_convergence(distribution, nsamples_range=100:200:2000)
    end
end

@testitem "MLE" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = InverseGamma(1, 0.1)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = InverseGamma(1, 1)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = InverseGamma(1, 10)
        @test test_projection_mle(distribution)
    end
end