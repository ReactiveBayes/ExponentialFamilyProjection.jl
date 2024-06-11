@testitem "Simple projection to `Chisq`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Chisq(1.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Chisq(4.0)
        @test_broken test_projection_convergence(distribution)
    end
end