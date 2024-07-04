@testitem "Simple projection to `Poisson`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Poisson(1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Poisson(10)
        @test test_projection_convergence(distribution, nsamples_range = 500:200:4000, niterations_nsamples = 700)
    end

    @testset let distribution = Poisson(0.5)
        @test test_projection_convergence(distribution)
    end
end


@testitem "Project `Geometric` to `Poisson`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Geometric(0.6)
        @test test_projection_convergence(distribution, to = Poisson)
    end

    @testset let distribution = Geometric(0.7)
        @test test_projection_convergence(distribution, to = Poisson)
    end
end
