@testitem "Simple projection to `Bernoulli`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Bernoulli(0.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.25)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.5)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.75)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Bernoulli(0.9)
        @test test_projection_convergence(distribution)
    end
end

@testitem "Project a product of `Bernoulli` and `Bernoulli` to `Bernoulli`" begin
    using BayesBase, ExponentialFamily, Distributions

    include("./projected_to_setuptests.jl")

    @testset let distribution = ProductOf(Bernoulli(0.3), Bernoulli(0.65))
        @test test_projection_convergence(distribution, to = Bernoulli, dims = (), conditioner = nothing)
    end

    @testset let distribution = ProductOf(Bernoulli(0.5), Bernoulli(0.95))
        @test test_projection_convergence(distribution, to = Bernoulli, dims = (), conditioner = nothing)
    end

end