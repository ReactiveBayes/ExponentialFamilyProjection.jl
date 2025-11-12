@testitem "Simple projection to `Bernoulli` with debug" begin
    using BayesBase, ExponentialFamily, Distributions, JET
    using ExponentialFamilyProjection

    include("../projected_to_setuptests.jl")

    function test_projection_with_debug(n_iterations, do_debug, pass_to_prj = false)
        distribution = Bernoulli(0.5)
        buf = IOBuffer()
        if do_debug
            debug = [DebugCost(io = buf), DebugDivider("\n"; io = buf)]
        else
            debug = missing
        end

        projection_parameters = ProjectionParameters(niterations = n_iterations)
        if pass_to_prj
            prj = ProjectedTo(
                Bernoulli,
                parameters = projection_parameters,
                kwargs = (debug = debug,),
            )
            project_to(prj, (x) -> logpdf(distribution, x))
        else
            prj = ProjectedTo(Bernoulli, parameters = projection_parameters)
            project_to(prj, (x) -> logpdf(distribution, x); debug = debug)
        end
        debug_string = String(take!(buf))
        if do_debug
            lines = split(debug_string, '\n')
            @test length(lines) == n_iterations + 2
            @test all(map(line -> occursin(r"f\(x\): -?\d+\.\d+", line), lines[2:(end-1)]))
        else
            @test debug_string == ""
        end
    end

    @testset "projections with debug" begin
        for n = 1:10
            test_projection_with_debug(n, true)
        end
    end

    @testset "projections without debug" begin
        for n = 1:10
            test_projection_with_debug(n, false)
        end
    end

    @testset "projections with debug passed through ProjectedTo" begin
        for n = 1:10
            test_projection_with_debug(n, true, true)
        end
    end
end
