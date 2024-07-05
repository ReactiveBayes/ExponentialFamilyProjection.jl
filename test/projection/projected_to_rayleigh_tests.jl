@testitem "Simple projection to `Rayleigh`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Rayleigh(1.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Rayleigh(10.0)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Rayleigh(0.5)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Rayleigh(20.7)
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Rayleigh(100.26)
        @test test_projection_convergence(distribution)
    end
end


@testitem "Project `Gamma` to `Rayleigh`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Gamma(4, 10)
        @test test_projection_convergence(
            distribution,
            to = Rayleigh
        )
    end

    @testset let distribution = Gamma(40, 10)
        @test_broken test_projection_convergence(
            distribution,
            to = Rayleigh
        )
    end
end

@testitem "Project `LogNormal` to `Rayleigh`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = LogNormal(0.1, 1)
        @test_broken test_projection_convergence(
            distribution,
            to = Rayleigh
        )
    end
end