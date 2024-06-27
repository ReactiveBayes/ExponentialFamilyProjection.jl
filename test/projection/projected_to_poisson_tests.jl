@testitem "Simple projection to `Poisson`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Poisson(1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Poisson(5.7)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Poisson(0.5)
        @test test_projection_convergence(distribution)
    end
end


@testitem "Project `Normal`,  to `Poisson`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Normal(2.5, 1)
        @test test_projection_convergence(distribution, to = Poisson)
    end

    @testset let distribution = Normal(2.2, 5.0)
        @test test_projection_convergence(distribution, to = Poisson)
    end
end
