@testitem "test nr of executions" begin
    using BayesBase

    include(".././projected_to_setuptests.jl")

    mutable struct CounterDist
        count::Int
    end
    CounterDist() = CounterDist(0)
    BayesBase.logpdf(d::CounterDist, x::Real) = begin
        d.count += 1
        return 0.0
    end
    for n_iterations in [3]
        for n_samples in [2]
            dist = CounterDist()
            prj = ProjectedTo(
                NormalMeanVariance,
                parameters = ProjectionParameters(
                    strategy = ExponentialFamilyProjection.ControlVariateStrategy(
                        nsamples = n_samples,
                    ),
                    niterations = n_iterations,
                ),
            )
            project_to(prj, (x) -> logpdf(dist, x))
            @test dist.count == n_iterations * n_samples
        end
    end
end