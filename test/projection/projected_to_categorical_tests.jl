@testitem "Simple projection to `Categorical`" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("./projected_to_setuptests.jl")

    @testset let distribution = Categorical([1/3, 1/3, 1/3])
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Categorical([1/4, 1/2, 1/4])
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Categorical([1/16, 1/2, 1/4, 3/16])
        @test test_projection_convergence(distribution)
    end

    @testset let distribution = Categorical([1/12, 1/3, 1/12, 1/6, 1/3])
        @test test_projection_convergence(distribution)
    end

 
end


@testitem "Project a product of `Categorical` and `Categorical` to `Categorical`" begin
    using BayesBase, ExponentialFamily, Distributions

    include("./projected_to_setuptests.jl")

    @testset let distribution = ProductOf(Categorical([1/3, 1/3, 1/3]), Categorical([1/8, 1/8, 3/4]))
        @test test_projection_convergence(distribution, to = Categorical, dims = (), conditioner = 3)
    end
end