@testitem "ControlVariateStrategy generic properties" begin
    using Random, Bumper, LinearAlgebra, Distributions, ExponentialFamily
    import ExponentialFamilyProjection:
        ControlVariateStrategy, getnsamples, getseed, getrng, getbuffer, getstate, prepare_state!

    @test ControlVariateStrategy() == ControlVariateStrategy()
    @test ControlVariateStrategy(nsamples = 100) == ControlVariateStrategy(nsamples = 100)
    @test ControlVariateStrategy(seed = 42) == ControlVariateStrategy(seed = 42)
    @test ControlVariateStrategy(rng = MersenneTwister(42)) == ControlVariateStrategy(rng = MersenneTwister(42))

    @test ControlVariateStrategy(nsamples = 50) !== ControlVariateStrategy(nsamples = 100)
    @test ControlVariateStrategy(seed = 41) !== ControlVariateStrategy(seed = 42)
    @test ControlVariateStrategy(rng = MersenneTwister(41)) !== ControlVariateStrategy(rng = MersenneTwister(42))

    @testset "nsamples" begin
        strategy = ControlVariateStrategy(nsamples = 100)

        @test getnsamples(strategy) === 100

        strategy = ControlVariateStrategy(nsamples = 200)

        @test getnsamples(strategy) === 200
    end

    @testset "seed" begin
        strategy = ControlVariateStrategy(seed = 42)

        @test getseed(strategy) === 42

        strategy = ControlVariateStrategy(seed = 24)

        @test getseed(strategy) === 24
    end

    @testset "rng" begin
        rng1 = MersenneTwister(42)
        rng2 = MersenneTwister(24)
        strategy = ControlVariateStrategy(rng = rng1)

        @test getrng(strategy) === rng1
        @test getrng(strategy) !== rng2

        strategy = ControlVariateStrategy(rng = rng2)

        @test getrng(strategy) !== rng1
        @test getrng(strategy) === rng2
    end

    @testset "state" begin 
        distributions = [Beta(5, 5), Chisq(10)]
        for dist in distributions
            ef = convert(ExponentialFamilyDistribution, Beta(5, 5))
            state1 = prepare_state!(ControlVariateStrategy(), (x) -> 1, ef, ())
            state2 = prepare_state!(ControlVariateStrategy(), (x) -> 1, ef, ())
            @test state1 == state2
            @test ControlVariateStrategy(state = state1) == ControlVariateStrategy(state = state2)

            state1 = prepare_state!(ControlVariateStrategy(), (x) -> 1, ef, (ef,))
            state2 = prepare_state!(ControlVariateStrategy(), (x) -> 1, ef, (ef,))
            
            @test state1 == state2 

            state1 = prepare_state!(ControlVariateStrategy(), (x) -> 1, ef, (ef, ef))
            state2 = prepare_state!(ControlVariateStrategy(), (x) -> 1, ef, (ef, ef))
            @test state1 == state2 
        
            state1 = prepare_state!(ControlVariateStrategy(), (x) -> 1, ef, ())
            state2 = prepare_state!(ControlVariateStrategy(), (x) -> 2, ef, ())
            @test state1 != state2
        end
    end
end

@testitem "ControlVariateStrategy prepare state" begin
    using JET, ExponentialFamily, Distributions, BayesBase, LinearAlgebra
    import ExponentialFamilyProjection:
        ControlVariateStrategy,
        ControlVariateStrategyState,
        prepare_state!,
        getsamples,
        getlogpdfs,
        getlogbasemeasures,
        getsufficientstatistics,
        getgradsamples

    dists = [
        NormalMeanVariance(0, 1),
        Gamma(1, 1),
        Beta(1, 1),
        MvNormalMeanCovariance([0.0, 0.0], [1.0 0.0; 0.0 1.0]),
    ]

    for dist in dists
        targetfn1 = let dist = dist
            (x) -> logpdf(dist, x)
        end

        targetfn2 = BayesBase.InplaceLogpdf(let dist = dist
            (out, x) -> logpdf(dist, x)
        end)

        for targetfn in [targetfn1, targetfn2], nsamples in (100, 200)
            ef = convert(ExponentialFamilyDistribution, dist)

            @testset "Empty state should create new state every time (no supplementary_η)" begin
                strategy = ControlVariateStrategy(nsamples = nsamples, state = nothing)

                @test_opt ignored_modules = (Base, LinearAlgebra, Distributions) prepare_state!(
                    strategy,
                    targetfn,
                    ef,
                    ()
                )

                state1 = prepare_state!(strategy, targetfn, ef, ())
                state2 = prepare_state!(strategy, targetfn, ef, ())

                @test state1 !== state2
                # `==` check that the content of the arrays are similar 
                # `!==` checks that the arrays are different in memory
                @test getsamples(state1) == getsamples(state2)
                @test getsamples(state1) !== getsamples(state2)
                @test getlogpdfs(state1) == getlogpdfs(state2)
                @test getlogpdfs(state1) !== getlogpdfs(state2)
                @test getsufficientstatistics(state1) == getsufficientstatistics(state2)
                @test getsufficientstatistics(state1) !== getsufficientstatistics(state2)
                @test getgradsamples(state1) == getgradsamples(state2)
                @test getgradsamples(state1) !== getgradsamples(state2)

                samples = rand(ef, nsamples)
                logpdfs = zeros(paramfloattype(ef), nsamples)
                logbasemeasures = zeros(paramfloattype(ef), nsamples)
                sufficientstatistics =
                    zeros(paramfloattype(ef), length(getnaturalparameters(ef)), nsamples)
                gradsamples = similar(sufficientstatistics)
                state3 = ControlVariateStrategyState(
                    samples = samples,
                    logpdfs = logpdfs,
                    logbasemeasures = logbasemeasures,
                    sufficientstatistics = sufficientstatistics,
                    gradsamples = gradsamples,
                )

                strategy_with_state =
                    ControlVariateStrategy(nsamples = nsamples, state = state3)
                state3_prepared = prepare_state!(strategy_with_state, targetfn, ef, ())

                @test state3 === state3_prepared
                @test getsamples(state3) === getsamples(state3_prepared)
                @test getlogpdfs(state3) === getlogpdfs(state3_prepared)
                @test getsufficientstatistics(state3) ===
                      getsufficientstatistics(state3_prepared)
                @test getlogbasemeasures(state3) === getlogbasemeasures(state3_prepared)
                @test getgradsamples(state3) === getgradsamples(state3_prepared)

                @test getsamples(state1) == getsamples(state3)
                @test getlogpdfs(state1) == getlogpdfs(state3)
                @test getlogbasemeasures(state1) == getlogbasemeasures(state3)
                @test getsufficientstatistics(state1) == getsufficientstatistics(state3)
                @test getgradsamples(state1) == getgradsamples(state3)
            end
        end
    end

end

@testitem "Gradient shouldn't depend on the scale of the `logpdf` when nsamples goes to infinity" begin
    import ExponentialFamilyProjection: CVICostGradientObjective, ControlVariateStrategy
    import ExponentialFamilyManifolds: get_natural_manifold
    using StableRNGs, ExponentialFamily, Manifolds, BayesBase

    dist = Beta(4, 6)
    targetfn1 = (x) -> logpdf(dist, x)
    targetfn2 = (x) -> logpdf(dist, x) - 1000

    strategy = ControlVariateStrategy(
        nsamples = 10^6
    )
    M = get_natural_manifold(Beta, ())

    rng = StableRNG(42)
    p = rand(rng, M)
    X1 = zero_vector(M, p)
    X2 = zero_vector(M, p)

    objective1 = CVICostGradientObjective(targetfn1, (), strategy, nothing)
    objective2 = CVICostGradientObjective(targetfn2, (), strategy, nothing)

    c1, X1 = objective1(M, X1, p)
    c2, X2 = objective2(M, X2, p)

    # `c2` is bigger, becase the `targetfn` is being subtracted from the optimized function
    @test c1 ≈ c2 - 1000
    @test X1 ≈ X2 rtol = 1e-4
end

@testitem "The CVI objective cost value with supplementary natural parameters should approximatly equal to just `ProductOf`" begin
    using ExponentialFamily, Distributions, BayesBase
    using Random, StableRNGs
    using Manopt, ExponentialFamilyManifolds, ExponentialFamilyProjection

    # This distribution are selected because their log base measure is zero
    # Once it's not zero the test will fail
    distributions = [
        (Bernoulli(0.7), Bernoulli(0.1)),
        (Bernoulli(0.1), Bernoulli(0.9)),
        (Bernoulli(0.9), Bernoulli(0.1)),
        (Beta(10, 10), Beta(3, 3)),
        (Beta(1, 1), Beta(0.1, 3)),
        (Gamma(1, 1), Gamma(10, 10)),
    ]

    for distribution in distributions
        @testset let left = distribution[1], right = distribution[2], nsamples = 2000, nseeds = 20
            dims = size(rand(left))
            
            typetag = ExponentialFamily.exponential_family_typetag(left)
            
            manifold = ExponentialFamilyManifolds.get_natural_manifold(
                typetag,
                dims,
                nothing
            )
    
            targetfn_part = (x) -> logpdf(left, x)
            targetfn_full = (x) -> logpdf(ProductOf(left, right), x)
            ef = convert(ExponentialFamilyDistribution, right)
            supplementary_ef = [getnaturalparameters(convert(ExponentialFamilyDistribution, right)),]

            seeds = rand(StableRNG(42), UInt, nseeds)
            point = rand(StableRNG(42), manifold)

            costs = map(seeds) do seed
                obj_part = ExponentialFamilyProjection.CVICostGradientObjective(
                    targetfn_part,
                    supplementary_ef,
                    ExponentialFamilyProjection.ControlVariateStrategy(nsamples = nsamples, seed = seed),
                    nothing
                )

                obj_full = ExponentialFamilyProjection.CVICostGradientObjective(
                    targetfn_full,
                    [],
                    ExponentialFamilyProjection.ControlVariateStrategy(nsamples = nsamples, seed = seed),
                    nothing
                )
                
                X2 = Manopt.zero_vector(manifold, point)
                X1 = Manopt.zero_vector(manifold, point)

                c1, _ = obj_part(manifold, X1, point)
                c2, _ = obj_full(manifold, X2, point)
            
                (c1, c2)
            end

            c1s = map((c) -> c[1], costs)
            c2s = map((c) -> c[2], costs)

            @test (mean(c1s) .+ logpartition(ef)) ≈ mean(c2s) rtol=1e-2
        end
    end
end