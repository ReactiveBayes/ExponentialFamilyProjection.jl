@testitem "Simple projection to `Dirichlet`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Dirichlet([1.0, 1.0])
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Dirichlet([0.5, 2.0])
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Dirichlet([2.0, 5.0])
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Dirichlet([3.14, 2.71, 6.81])
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Dirichlet([2.0, 5.0, 7.0, 0.5])
        @test test_projection_convergence(distribution)
    end

end

@testitem "MLE" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Dirichlet([1.0, 1.0])
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Dirichlet([0.5, 2.0])
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Dirichlet([2.0, 5.0])
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Dirichlet([3.14, 2.71, 6.81])
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Dirichlet([2.0, 5.0, 7.0, 0.5])
        @test test_projection_mle(distribution)
    end

end
