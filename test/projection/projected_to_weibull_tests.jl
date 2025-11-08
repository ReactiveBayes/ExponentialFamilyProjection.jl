@testitem "Simple projection to `Weibull`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Weibull()
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Weibull(0.1, 2.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Weibull(3.14, 2.71)
        @test test_projection_convergence(distribution)
    end
end

#Because Weibull with k = 1 is Exponential distribution, we fix the conditioner to 1.
@testitem "Project `Exponential` to `Weibull`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Exponential(0.1)
        @test test_projection_convergence(distribution, to = Weibull, conditioner = 1.0)
    end

    @testset let distribution = Exponential(0.1)
        @test_throws AssertionError test_projection_convergence(
            distribution,
            to = Weibull,
            conditioner = -1.0,
        )
    end

    @testset let distribution = Exponential(10.1)
        @test test_projection_convergence(distribution, to = Weibull, conditioner = 1.0)
    end

    @testset let distribution = Exponential(100.32)
        @test test_projection_convergence(distribution, to = Weibull, conditioner = 1.0)
    end
end

@testitem "Project `Rayleigh` to `Weibull`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Rayleigh(0.1)
        @test test_projection_convergence(distribution, to = Weibull, conditioner = 2.0)
    end

    @testset let distribution = Rayleigh(0.1)
        @test_throws AssertionError test_projection_convergence(
            distribution,
            to = Weibull,
            conditioner = -2.0,
        )
    end

    @testset let distribution = Rayleigh(10.1)
        @test test_projection_convergence(distribution, to = Weibull, conditioner = 2.0)
    end

    @testset let distribution = Rayleigh(100.32)
        @test test_projection_convergence(distribution, to = Weibull, conditioner = 2.0)
    end
end


@testitem "MLE" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Weibull()
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Weibull(0.1, 2.0)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Weibull(3.14, 2.71)
        @test test_projection_mle(distribution)
    end
end
