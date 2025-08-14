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

@testitem "BonnetStrategy projection convergence for `Normal`" begin
    using BayesBase, ExponentialFamily, Distributions
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset "BonnetStrategy nsamples convergence" begin
        @testset let distribution = NormalMeanVariance(1.0, 1.0)
            @test test_bonnet_projection_convergence(distribution)
        end

        @testset let distribution = NormalMeanVariance(-5.0, 0.5)
            @test test_bonnet_projection_convergence(distribution)
        end

        @testset let distribution = NormalMeanVariance(0.0, 2.0)
            @test test_bonnet_projection_convergence(distribution)
        end

        @testset let distribution = NormalMeanVariance(-3.14, 2.71)
            @test test_bonnet_projection_convergence(distribution)
        end
    end

    @testset "BonnetStrategy niterations convergence" begin
        @testset let distribution = NormalMeanVariance(2.0, 1.5)
            @test test_bonnet_niterations_convergence(distribution)
        end

        @testset let distribution = NormalMeanVariance(0.0, 1.0)
            @test test_bonnet_niterations_convergence(distribution)
        end
    end
end

@testitem "GaussNewton projection convergence for `Normal`" begin
    using BayesBase, ExponentialFamily, Distributions
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset "GaussNewton niterations convergence" begin
        @testset let distribution = NormalMeanVariance(1.0, 1.0)
            @test test_gaussnewton_projection_convergence(distribution)
        end

        @testset let distribution = NormalMeanVariance(-5.0, 0.5)
            @test test_gaussnewton_projection_convergence(distribution)
        end

        @testset let distribution = NormalMeanVariance(0.0, 2.0)
            @test test_gaussnewton_projection_convergence(distribution)
        end

        @testset let distribution = NormalMeanVariance(-3.14, 2.71)
            @test test_gaussnewton_projection_convergence(distribution)
        end
    end
end

@testitem "BonnetStrategy projection convergence for multivariate `Normal`" begin
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset "BonnetStrategy multivariate convergence" begin
        @testset let distribution = MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 1.0])
            @test test_bonnet_projection_convergence(
                distribution,
                nsamples_range = 500:100:2000,
                nsamples_niterations = 1000,
                nsamples_required_accuracy = 1e-1
            )
        end

        @testset let distribution = MvNormalMeanCovariance(zeros(3), Matrix(I, 3, 3))
            @test test_bonnet_projection_convergence(
                distribution,
                nsamples_range = 500:100:2000,
                nsamples_niterations = 1000,
                nsamples_required_accuracy = 1e-1
            )
        end

        @testset let distribution = MvNormalMeanCovariance([0.5, -1.5], [1.5 0.3; 0.3 0.8])
            @test test_bonnet_projection_convergence(
                distribution,
                nsamples_range = 500:100:2000,
                nsamples_niterations = 1000,
                nsamples_required_accuracy = 1e-1
            )
        end

        @testset let distribution = MvNormalMeanCovariance(10randn(StableRNG(42), 4), 10rand(StableRNG(43), 4))
                @test test_bonnet_projection_convergence(distribution, niterations_range = 500:100:2000)
        end
    end
end

@testitem "GaussNewton projection convergence for multivariate `Normal`" begin
    using BayesBase, ExponentialFamily, Distributions, LinearAlgebra
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset "GaussNewton multivariate convergence" begin
        @testset let distribution = MvNormalMeanCovariance([1.0, 2.0], [2.0 0.5; 0.5 1.0])
            @test test_gaussnewton_projection_convergence(distribution)
        end

        @testset let distribution = MvNormalMeanCovariance(zeros(3), Matrix(I, 3, 3))
            @test test_gaussnewton_projection_convergence(distribution)
        end

        @testset let distribution = MvNormalMeanCovariance([0.5, -1.5], [1.5 0.3; 0.3 0.8])
            @test test_gaussnewton_projection_convergence(distribution)
        end
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

@testitem "MLE: NormalMeanVariance" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")
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
end

@testitem "MLE: MvNormalMeanScalePrecision" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection, LinearAlgebra

    include("./projected_to_setuptests.jl")

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