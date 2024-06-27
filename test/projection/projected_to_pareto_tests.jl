@testitem "Simple projection to `Pareto`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Pareto(1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Pareto(1.7, 5.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Pareto(0.5,2)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Pareto(10.1,20.0)
        @test test_projection_convergence(distribution)
    end
end



