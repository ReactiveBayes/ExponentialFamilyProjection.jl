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

    @testset let distribution = Normal(-3.14, 0.01)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Normal(-3.14, 0.0001)
        @test_broken test_projection_convergence(distribution)
    end
end

@testitem "Project a product of `Normal` and `Normal` to `Normal`" begin
    using BayesBase, ExponentialFamily, Distributions

    include("./projected_to_setuptests.jl")

    @testset let distribution = ProductOf(Normal(3.14, 7.13), Normal(-0.87, 15.4))
        @test test_projection_convergence(
            distribution,
            to = NormalMeanVariance,
            dims = (),
            conditioner = nothing,
        )
    end

    @testset let distribution = ProductOf(Normal(0.5, 1.0), Normal(0.95, 2.0))
        @test test_projection_convergence(
            distribution,
            to = NormalMeanVariance,
            dims = (),
            conditioner = nothing,
        )
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
            niterations_stepsize = ConstantStepsize(0.1),
        )
    end
end

@testitem "Project a product of `MvNormal` and `MvNormal` to `MvNormal`" begin
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    include("./projected_to_setuptests.jl")

    @testset let distribution = ProductOf(
            MvNormalMeanCovariance(ones(2), Matrix(Diagonal(ones(2)))),
            MvNormalMeanCovariance(ones(2), Matrix(Diagonal(ones(2)))),
        )
        @test test_projection_convergence(
            distribution,
            to = MvNormalMeanCovariance,
            dims = (2, ),
            conditioner = nothing,
        )
    end

end