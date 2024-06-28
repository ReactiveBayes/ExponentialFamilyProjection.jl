@testitem "Simple projection to `Gamma`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Gamma(1, 0.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Gamma(1, 1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Gamma(1, 10)
        @test test_projection_convergence(distribution)
    end
end

@testitem "Project `Normal` to `Gamma`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Normal(4, 10)
        @test test_projection_convergence(
            distribution,
            to = Gamma,
            niterations_required_accuracy = 5e-1,
            nsamples_required_accuracy = 5e-1,
        )
    end
end