@testitem "Simple projection to `Normal`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Normal(1, 1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Normal(-5, 0.5)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Normal(1, 10)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Normal(-10, 1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Normal(-3.14, 2.71)
        @test test_projection_convergence(distribution)
    end
end

@testitem "Simple projection to `MvNormal`" begin
    using BayesBase, ExponentialFamily, Distributions, JET, LinearAlgebra, Manopt
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = MvNormalMeanCovariance(ones(2), Matrix(Diagonal(ones(2))))
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = MvNormalMeanCovariance(
            [3.14, 2.71, -6.89],
            [1.0 -0.1 -0.2; -0.1 3.0 -0.4; -0.2 -0.4 9.0],
        )
        @test test_projection_convergence(distribution)
    end

    @testset let distribution =
            MvNormalMeanCovariance(10randn(StableRNG(42), 4), 10rand(StableRNG(43), 4))
        @test test_projection_convergence(
            distribution,
            niterations_range = 500:100:2000,
            niterations_stepsize = ConstantStepsize(0.01),
        )
    end
end