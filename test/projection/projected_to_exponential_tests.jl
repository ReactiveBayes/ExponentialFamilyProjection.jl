@testitem "Simple projection to `Exponential`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Exponential()
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Exponential(0.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Exponential(0.9)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Exponential(2.9)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Exponential(7.41)
        @test test_projection_convergence(distribution)
    end
end

@testitem "MLE" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Exponential()
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Exponential(0.1)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Exponential(0.9)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Exponential(2.9)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Exponential(7.41)
        @test test_projection_mle(distribution)
    end
end