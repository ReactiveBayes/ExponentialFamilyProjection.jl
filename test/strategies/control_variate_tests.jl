@testitem "ControlVariateStrategy generic properties" begin
    using Random,
        BayesBase,
        Bumper,
        LinearAlgebra,
        Distributions,
        ExponentialFamily,
        ExponentialFamilyManifolds
    import ExponentialFamilyProjection:
        ControlVariateStrategy,
        ProjectionParameters,
        getnsamples,
        create_state!,
        prepare_state!

    @test ControlVariateStrategy() == ControlVariateStrategy()
    @test ControlVariateStrategy(nsamples = 100) == ControlVariateStrategy(nsamples = 100)
    @test ControlVariateStrategy(nsamples = 50) !== ControlVariateStrategy(nsamples = 100)


    @testset "nsamples" begin
        strategy = ControlVariateStrategy(nsamples = 100)

        @test getnsamples(strategy) === 100

        strategy = ControlVariateStrategy(nsamples = 200)

        @test getnsamples(strategy) === 200
    end

    @testset "create_state!" begin
        distributions = [Beta(5, 5), Chisq(10)]
        parameters = ProjectionParameters()
        for dist in distributions
            ef = convert(ExponentialFamilyDistribution, Beta(5, 5))
            T = ExponentialFamily.exponential_family_typetag(ef)
            d = size(mean(ef))
            c = getconditioner(ef)
            M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)
            arg = (x) -> 1
            state1 = create_state!(ControlVariateStrategy(), M, parameters, arg, ef, ())
            state2 = create_state!(ControlVariateStrategy(), M, parameters, arg, ef, ())
            @test state1 == state2

            state1 = create_state!(ControlVariateStrategy(), M, parameters, arg, ef, (ef,))
            state2 = create_state!(ControlVariateStrategy(), M, parameters, arg, ef, (ef,))

            @test state1 == state2

            state1 =
                create_state!(ControlVariateStrategy(), M, parameters, arg, ef, (ef, ef))
            state2 =
                create_state!(ControlVariateStrategy(), M, parameters, arg, ef, (ef, ef))
            @test state1 == state2
        end
    end
end

@testitem "ControlVariateStrategy prepare state" begin
    using JET,
        ExponentialFamily,
        Distributions,
        BayesBase,
        LinearAlgebra,
        StableRNGs,
        ExponentialFamilyManifolds
    import ExponentialFamilyProjection:
        ControlVariateStrategy,
        ControlVariateStrategyState,
        create_state!,
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
        Chisq(30.0),
    ]

    for dist in dists

        targetfn1 = let dist = dist
            (x) -> logpdf(dist, x)
        end

        targetfn2 = BayesBase.InplaceLogpdf(let dist = dist
            (out, x) -> logpdf(dist, x)
        end)

        for targetfn in [targetfn1, targetfn2],
            nsamples in (100, 200),
            supplementary in ((), (dist,))

            ef = convert(ExponentialFamilyDistribution, dist)
            T = ExponentialFamily.exponential_family_typetag(ef)
            d = size(mean(ef))
            c = getconditioner(ef)
            M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)

            supplementary_η = map(
                d -> getnaturalparameters(convert(ExponentialFamilyDistribution, d)),
                supplementary,
            )

            @testset "Empty state should create new state every time (no supplementary_η)" begin
                rng = StableRNG(42)
                parameters = ProjectionParameters(rng = rng)
                strategy = ControlVariateStrategy(nsamples = nsamples)

                @test_opt ignored_modules = (Base, LinearAlgebra, Distributions) create_state!(
                    strategy,
                    M,
                    parameters,
                    targetfn,
                    ef,
                    supplementary_η,
                )

                @test_opt ignored_modules = (Base, LinearAlgebra, Distributions) ExponentialFamilyProjection.prepare_samples_container(
                    rng,
                    ef,
                    nsamples,
                    supplementary_η,
                )
                @test_opt ignored_modules = (Base, LinearAlgebra, Distributions) ExponentialFamilyProjection.prepare_logpdfs_container(
                    rng,
                    ef,
                    nsamples,
                    supplementary_η,
                )
                @test_opt ignored_modules = (Base, LinearAlgebra, Distributions) ExponentialFamilyProjection.prepare_logbasemeasures_container(
                    rng,
                    ef,
                    nsamples,
                    supplementary_η,
                )
                @test_opt ignored_modules = (Base, LinearAlgebra, Distributions) ExponentialFamilyProjection.prepare_sufficientstatistics_container(
                    rng,
                    ef,
                    nsamples,
                    supplementary_η,
                )
                @test_opt ignored_modules = (Base, LinearAlgebra, Distributions) ExponentialFamilyProjection.prepare_gradsamples_container(
                    rng,
                    ef,
                    nsamples,
                    supplementary_η,
                )

                state1 =
                    create_state!(strategy, M, parameters, targetfn, ef, supplementary_η)
                state2 =
                    create_state!(strategy, M, parameters, targetfn, ef, supplementary_η)

                @test state1 == state2
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

                if isbasemeasureconstant(ef) === ConstantBaseMeasure()
                    @test getlogbasemeasures(state1) === getlogbasemeasures(state2)
                else
                    @test getlogbasemeasures(state1) !== getlogbasemeasures(state2)
                end

                samples = rand(ef, nsamples)
                logpdfs = zeros(paramfloattype(ef), nsamples)
                logbasemeasures = if isbasemeasureconstant(ef) === ConstantBaseMeasure()
                    fill((1 - length(supplementary)) * log(basemeasure(ef, rand(ef))), nsamples)
                else
                    zeros(paramfloattype(ef), nsamples)
                end
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

                strategy = ControlVariateStrategy(nsamples = nsamples)
                state3_prepared = prepare_state!(
                    strategy,
                    state3,
                    M,
                    parameters,
                    targetfn,
                    ef,
                    supplementary_η,
                )

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
    import ExponentialFamilyProjection:
        ProjectionCostGradientObjective, ControlVariateStrategy, create_state!
    import ExponentialFamilyManifolds: get_natural_manifold
    using StableRNGs, ExponentialFamily, Manifolds, BayesBase

    dist = Beta(4, 6)
    targetfn1 = (x) -> logpdf(dist, x)
    targetfn2 = (x) -> logpdf(dist, x) - 1000

    strategy = ControlVariateStrategy(nsamples = 10^6)
    parameters = ProjectionParameters()
    M = get_natural_manifold(Beta, ())

    rng = StableRNG(42)
    p = rand(rng, M)
    X1 = zero_vector(M, p)
    X2 = zero_vector(M, p)

    state1 = create_state!(
        strategy,
        M,
        parameters,
        targetfn1,
        convert(ExponentialFamilyDistribution, M, p),
        (),
    )
    state2 = create_state!(
        strategy,
        M,
        parameters,
        targetfn2,
        convert(ExponentialFamilyDistribution, M, p),
        (),
    )

    objective1 = ProjectionCostGradientObjective(
        parameters,
        targetfn1,
        (),
        strategy,
        state1,
        nothing,
    )
    objective2 = ProjectionCostGradientObjective(
        parameters,
        targetfn2,
        (),
        strategy,
        state2,
        nothing,
    )

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
        @testset let left = distribution[1],
            right = distribution[2],
            nsamples = 2000,
            nseeds = 20

            dims = size(rand(left))

            typetag = ExponentialFamily.exponential_family_typetag(left)

            manifold =
                ExponentialFamilyManifolds.get_natural_manifold(typetag, dims, nothing)


            targetfn_part = (x) -> logpdf(left, x)
            targetfn_full = (x) -> logpdf(ProductOf(left, right), x)
            ef = convert(ExponentialFamilyDistribution, right)
            supplementary_ef =
                [getnaturalparameters(convert(ExponentialFamilyDistribution, right))]

            seeds = rand(StableRNG(42), UInt, nseeds)
            point = rand(StableRNG(42), manifold)

            costs = map(seeds) do seed
                parameters = ProjectionParameters(seed = seed)
                strategy = ExponentialFamilyProjection.ControlVariateStrategy(
                    nsamples = nsamples,
                )
                state_part = ExponentialFamilyProjection.create_state!(
                    strategy,
                    manifold,
                    parameters,
                    targetfn_part,
                    convert(ExponentialFamilyDistribution, manifold, point),
                    supplementary_ef,
                )
                obj_part = ExponentialFamilyProjection.ProjectionCostGradientObjective(
                    parameters,
                    targetfn_part,
                    supplementary_ef,
                    strategy,
                    state_part,
                    nothing,
                )

                state_full = ExponentialFamilyProjection.create_state!(
                    strategy,
                    manifold,
                    parameters,
                    targetfn_full,
                    convert(ExponentialFamilyDistribution, manifold, point),
                    [],
                )
                obj_full = ExponentialFamilyProjection.ProjectionCostGradientObjective(
                    parameters,
                    targetfn_full,
                    [],
                    strategy,
                    state_full,
                    nothing,
                )

                X2 = Manopt.zero_vector(manifold, point)
                X1 = Manopt.zero_vector(manifold, point)

                c1, _ = obj_part(manifold, X1, point)
                c2, _ = obj_full(manifold, X2, point)

                (c1, c2)
            end

            c1s = map((c) -> c[1], costs)
            c2s = map((c) -> c[2], costs)

            @test (mean(c1s) .+ logpartition(ef)) ≈ mean(c2s) rtol = 1e-2
        end
    end
end

@testitem "`ControlVariateStrategy` should fail if given a list of samples instead of a function" begin
    using ExponentialFamily

    prj = ProjectedTo(
        Beta;
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.ControlVariateStrategy(),
        ),
    )

    @test_throws "The `ControlVariateStrategy` requires the projection argument to be a callable object (e.g. `Function`)." project_to(
        prj,
        [0.5],
    )

end