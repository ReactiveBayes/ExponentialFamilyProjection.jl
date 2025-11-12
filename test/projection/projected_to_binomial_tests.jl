@testitem "Simple projection to `Binomial`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Binomial(10, 0.5)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Binomial(20, 0.1)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Binomial(2, 0.9)
        @test test_projection_convergence(distribution)
    end


end


@testitem "Project `Poisson` to `Binomial`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Poisson(0.6)
        @test test_projection_convergence(distribution, to = Binomial, conditioner = 5)
    end

    @testset let distribution = Poisson(5.3)
        @test test_projection_convergence(distribution, to = Binomial, conditioner = 20)
    end

    @testset let distribution = Poisson(5.3)
        @test_throws AssertionError test_projection_convergence(
            distribution,
            to = Binomial,
            conditioner = -20,
        )
    end
end



@testitem "Project `Bernoulli` to `Binomial`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Bernoulli(0.6)
        @test test_projection_convergence(distribution, to = Binomial, conditioner = 1)
    end

    @testset let distribution = Bernoulli(0.6)
        @test_throws AssertionError test_projection_convergence(
            distribution,
            to = Binomial,
            conditioner = -1,
        )
    end

    @testset let distribution = Bernoulli(0.3)
        @test test_projection_convergence(distribution, to = Binomial, conditioner = 1)
    end

    @testset let distribution = Bernoulli(0.99)
        @test test_projection_convergence(distribution, to = Binomial, conditioner = 1)
    end

    @testset let distribution = Bernoulli(0.001)
        @test test_projection_convergence(distribution, to = Binomial, conditioner = 1)
    end
end

@testitem "MLE" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Binomial(10, 0.5)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Binomial(20, 0.1)
        @test test_projection_mle(distribution)
    end

    @testset let distribution = Binomial(2, 0.9)
        @test test_projection_mle(distribution)
    end


end
