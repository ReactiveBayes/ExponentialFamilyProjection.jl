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
        @test test_projection_convergence(distribution, niterations_range = 500:100:2000)
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
            dims = (2,),
            conditioner = nothing,
        )
    end

end

@testitem "Project a product of `MvNormalMeanScalePrecision` and `MvNormalMeanScalePrecision` to `MvNormalMeanScalePrecision`" begin
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra

    include("./projected_to_setuptests.jl")

    @testset let distribution = ProductOf(
            MvNormalMeanScalePrecision(ones(2), 2),
            MvNormalMeanScalePrecision(ones(2), 3),
        )
        @test test_projection_convergence(
            distribution,
            to = MvNormalMeanScalePrecision,
            dims = (2,),
            conditioner = nothing,
        )
    end

    @testset let distribution = ProductOf(
            MvNormalMeanScalePrecision(ones(8), 2),
            MvNormalMeanScalePrecision(ones(8), 3),
        )
        @test test_projection_convergence(
            distribution,
            to = MvNormalMeanScalePrecision,
            dims = (8,),
            conditioner = nothing,
        )
    end

    @testset let distribution = ProductOf(
        MvNormalMeanScalePrecision(ones(20), 2),
        MvNormalMeanScalePrecision(ones(20), 3),
    )
        @test test_projection_convergence(
            distribution,
            to = MvNormalMeanScalePrecision,
            dims = (20,),
            conditioner = nothing,
            nsamples_niterations = 6000,
            nsamples_range = 1000:1000:6000,
            niterations_range = 400:100:1000,
            nsamples_required_accuracy=0.3,
            niterations_required_accuracy=0.3
        )
    end

end

@testitem "MLE" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection, LinearAlgebra

    include("./projected_to_setuptests.jl")

    @testset let distribution = Normal(1, 1)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Normal(-5, 0.5)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Normal(1, 10)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Normal(-10, 1)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Normal(-3.14, 2.71)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = MvNormalMeanScalePrecision(ones(2), 2)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = MvNormalMeanCovariance(ones(2), Matrix(Diagonal(ones(2))))
        @test test_projection_mle(distribution)
    end

    @testset let distribution = MvNormalMeanCovariance(
            [3.14, 2.71, -6.89],
            [1.0 -0.1 -0.2; -0.1 3.0 -0.4; -0.2 -0.4 9.0],
        )
        @test test_projection_mle(distribution)
    end

    @testset let distribution =
            MvNormalMeanCovariance(10randn(StableRNG(42), 4), 10rand(StableRNG(43), 4))
        @test test_projection_mle(distribution)
    end

end