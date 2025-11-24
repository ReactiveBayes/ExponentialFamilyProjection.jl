@testitem "ClosedFormStrategy generic properties" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    import ExponentialFamilyProjection: get_nsamples

    strategy = ClosedFormStrategy()

    @test strategy isa ClosedFormStrategy
    @test get_nsamples(strategy) == 0
end

@testitem "ClosedFormStrategy create_state!" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using ExponentialFamilyManifolds
    using ExponentialFamily
    using Distributions
    import ExponentialFamilyProjection: create_state!, ProjectionParameters

    distributions = [Beta(5, 5), NormalMeanVariance(0, 1), Gamma(2, 2)]
    parameters = ProjectionParameters()

    for dist in distributions
        ef = convert(ExponentialFamilyDistribution, dist)
        T = ExponentialFamily.exponential_family_typetag(ef)
        d = size(mean(ef))
        c = getconditioner(ef)
        M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)

        # Target wrapped in Logpdf
        target = Logpdf(dist)

        # Test without supplementary parameters
        state1 = create_state!(ClosedFormStrategy(), M, parameters, target, ef, ())
        state2 = create_state!(ClosedFormStrategy(), M, parameters, target, ef, ())

        @test state1.target === target
        @test state2.target === target
        @test state1 !== state2  # Different objects in memory

        # Test with supplementary parameters
        state3 = create_state!(ClosedFormStrategy(), M, parameters, target, ef, (ef,))
        state4 = create_state!(ClosedFormStrategy(), M, parameters, target, ef, (ef,))

        @test state3.target === target
        @test state4.target === target
        @test state3 !== state4

        # Test with multiple supplementary parameters
        state5 = create_state!(ClosedFormStrategy(), M, parameters, target, ef, (ef, ef))
        state6 = create_state!(ClosedFormStrategy(), M, parameters, target, ef, (ef, ef))

        @test state5.target === target
        @test state6.target === target
        @test state5 !== state6
    end
end

@testitem "ClosedFormStrategy prepare_state!" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using ExponentialFamilyManifolds
    using ExponentialFamily
    using Distributions
    import ExponentialFamilyProjection: create_state!, prepare_state!, ProjectionParameters

    distributions = [NormalMeanVariance(0, 1), Gamma(2, 2), Beta(3, 3)]

    for dist in distributions
        target_dist = dist
        target = Logpdf(target_dist)

        ef = convert(ExponentialFamilyDistribution, dist)
        T = ExponentialFamily.exponential_family_typetag(ef)
        d = size(mean(ef))
        c = getconditioner(ef)
        M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)

        strategy = ClosedFormStrategy()
        parameters = ProjectionParameters()

        # Create initial state
        state1 = create_state!(strategy, M, parameters, target, ef, ())

        # prepare_state! should return the same state object
        state2 = prepare_state!(strategy, state1, M, parameters, target, ef, ())

        @test state1 === state2
        @test state1.target === state2.target

        # Test with supplementary parameters
        supplementary_η = (getnaturalparameters(ef),)
        state3 = create_state!(strategy, M, parameters, target, ef, supplementary_η)
        state4 =
            prepare_state!(strategy, state3, M, parameters, target, ef, supplementary_η)

        @test state3 === state4
    end
end

@testitem "ClosedFormStrategy should fail if given a list of samples instead of a function" begin
    using ExponentialFamily
    using ClosedFormExpectations

    prj = ProjectedTo(
        Beta;
        parameters = ProjectionParameters(strategy = ClosedFormStrategy()),
    )

    # ClosedFormStrategy doesn't explicitly reject arrays, but it won't work properly
    # The extension's preprocess_strategy_argument will keep the array as-is
    # and then the compute_gradient! will fail when trying to use it
    # This is the expected behavior - it will error during execution

    samples = [0.5, 0.6, 0.7]

    # This should fail during the projection, not during preprocessing
    @test_throws Exception project_to(prj, samples)
end


@testitem "ClosedFormStrategy argument preprocessing for Distribution in closure" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using Distributions
    import ExponentialFamilyProjection: preprocess_strategy_argument

    strategy = ClosedFormStrategy()
    dist1 = Normal(0, 1)

    # Test extraction of Distribution from a closure (simulating RxInfer behavior)
    # The closure captures a Distribution, and preprocess should extract it
    closure_with_dist = let d = dist1
        (x) -> logpdf(d, x)
    end

    result_strat, result_arg = preprocess_strategy_argument(strategy, closure_with_dist)
    @test result_strat === strategy

    # The function should extract the captured Distribution and wrap it in Logpdf
    @test result_arg isa Logpdf
    @test result_arg.dist === dist1
end

@testitem "ClosedFormStrategy argument preprocessing for ProductOf in closure" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using Distributions
    using BayesBase
    import ExponentialFamilyProjection: preprocess_strategy_argument
    import BayesBase: ProductOf

    strategy = ClosedFormStrategy()

    # Test extraction of ProductOf from a closure (RxInfer use case)
    left = Beta(10, 10)
    right = Beta(3, 3)
    prod = ProductOf(left, right)

    closure_with_product = let p = prod
        (x) -> logpdf(p, x)
    end

    result_strat, result_arg = preprocess_strategy_argument(strategy, closure_with_product)
    @test result_strat === strategy

    # The function should extract the captured ProductOf and wrap it in Logpdf
    @test result_arg isa Logpdf
    @test result_arg.dist === prod
end

@testitem "ClosedFormStrategy argument preprocessing for plain function should error" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    import ExponentialFamilyProjection: preprocess_strategy_argument

    strategy = ClosedFormStrategy()

    # Plain function without captured variables should throw an error
    # because ClosedFormStrategy needs to extract Distribution/ProductOf from closure
    fn = (x) -> x^2

    @test_throws "`ClosedFormStrategy` requires a function that captures a `Distribution` or `ProductOf` in its closure" preprocess_strategy_argument(
        strategy,
        fn,
    )
end

@testitem "ClosedFormStrategy argument preprocessing for direct Distribution" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using Distributions
    import ExponentialFamilyProjection: preprocess_strategy_argument

    strategy = ClosedFormStrategy()
    dist1 = Normal(0, 1)

    # Case 4: Distribution directly (non-Function argument)
    result_strat, result_arg = preprocess_strategy_argument(strategy, dist1)
    @test result_strat === strategy
    @test result_arg isa ClosedFormExpectations.Logpdf
    @test result_arg.dist == dist1
end

@testitem "ClosedFormStrategy cost computation" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using ExponentialFamily
    using Distributions
    using ExponentialFamilyManifolds
    using StableRNGs
    import ExponentialFamilyProjection: compute_cost, create_state!, ProjectionParameters

    strategy = ClosedFormStrategy()

    # Target: Normal(0, 1)
    target_dist = Normal(0.0, 1.0)
    target = Logpdf(target_dist)

    # Variational: Normal(1, 1)
    # KL(q || p) where p=N(0,1), q=N(1,1) (variances equal)
    # KL = 0.5 * ( (μ1-μ2)^2/σ^2 ) = 0.5 * (1-0)^2/1 = 0.5

    # Manifold setup
    M = ExponentialFamilyManifolds.get_natural_manifold(NormalMeanVariance, ())
    q_dist = NormalMeanVariance(1.0, 1.0)
    ef = convert(ExponentialFamilyDistribution, q_dist)
    η = getnaturalparameters(ef)

    # Create state
    parameters = ProjectionParameters()
    state = create_state!(strategy, M, parameters, target, ef, ())

    # Dummy args for cost (not all used by ClosedFormStrategy's compute_cost)
    logp = logpartition(ef)
    gradlogp = gradlogpartition(ef)
    inv_fisher = inv(fisherinformation(ef))

    cost = compute_cost(M, strategy, state, η, logp, gradlogp, inv_fisher)

    # compute_cost returns: -entropy(q) - E_q[log p]
    # KL(q||p) = E_q[log q] - E_q[log p] = -entropy(q) - E_q[log p]
    # So it should approximately equal KL
    @test cost ≈ 0.5 atol = 1e-6
end

@testitem "ClosedFormStrategy gradient computation" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using ExponentialFamily
    using Distributions
    using ExponentialFamilyManifolds
    using LinearAlgebra
    using StableRNGs
    import ExponentialFamilyProjection:
        compute_gradient!, create_state!, ProjectionParameters

    strategy = ClosedFormStrategy()

    # Simple test: project Normal to Normal
    target_dist = Normal(2.0, 1.0)
    target = Logpdf(target_dist)

    # Start from a different point
    q_dist = NormalMeanVariance(0.0, 1.0)
    ef = convert(ExponentialFamilyDistribution, q_dist)

    M = ExponentialFamilyManifolds.get_natural_manifold(NormalMeanVariance, ())
    η = getnaturalparameters(ef)

    parameters = ProjectionParameters()
    state = create_state!(strategy, M, parameters, target, ef, ())

    # Compute gradient
    X = similar(η)
    logp = logpartition(ef)
    gradlogp = gradlogpartition(ef)
    inv_fisher = inv(fisherinformation(ef))

    X_result = compute_gradient!(M, strategy, state, X, η, logp, gradlogp, inv_fisher)

    @test X_result === X
    @test length(X) == length(η)
    @test all(isfinite.(X))

    # The gradient should point towards the target
    # Since target is at μ=2, gradient should push η in that direction
end


@testitem "LogGamma projected to Normal" begin
    using BayesBase
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily
    using StableRNGs
    using LinearAlgebra

    target_dist = LogGamma(20.0, 1.0)
    target = Logpdf(target_dist)

    # Create strategy
    strategy = ClosedFormStrategy()

    # Project
    result = project_to(
        ProjectedTo(NormalMeanVariance),
        target;
        strategy = strategy,
        parameters = ProjectionParameters(niterations = 50, tolerance = 1e-5),
    )

    @test result isa NormalMeanVariance
    μ, v = mean(result), var(result)

    # Mode of LogGamma(α, β) is at x = log(α*β)
    # Here α=20, β=1. Mode at log(20) ≈ 3.0
    @test 2.0 < μ < 4.0
end

@testitem "LogNormal projected to Gamma" begin
    using BayesBase
    using ExponentialFamilyProjection
    using ExponentialFamilyProjection: ControlVariateStrategy
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily
    using StableRNGs
    using LinearAlgebra

    # Target: LogNormal(μ=1.0, σ=0.5)
    target_dist = LogNormal(1.0, 0.5)
    target = Logpdf(target_dist)

    # Initial: Gamma(2.0, 2.0)
    initial_dist = Gamma(2.0, 2.0)

    strategy = ClosedFormStrategy()

    result = project_to(
        ProjectedTo(Gamma),
        target;
        strategy = strategy,
        initial_point = initial_dist,
        parameters = ProjectionParameters(niterations = 50, tolerance = 1e-5),
    )

    @test result isa GammaDistributionsFamily

    # Check if it ran without error and produced valid parameters
    @test shape(result) > 0
    @test scale(result) > 0

    # Comparison with ControlVariateStrategy
    cv_strategy = ControlVariateStrategy(nsamples = 500)
    result_cv = project_to(
        ProjectedTo(Gamma),
        target;
        strategy = cv_strategy,
        initial_point = initial_dist,
        parameters = ProjectionParameters(niterations = 50, tolerance = 1e-4),
    )

    # Should be close
    @test isapprox(mean(result), mean(result_cv), rtol = 0.1)
end

@testitem "ClosedFormStrategy vs ControlVariateStrategy: Speed and Accuracy" begin
    using BayesBase
    using ExponentialFamilyProjection
    using ExponentialFamilyProjection: ControlVariateStrategy
    using ClosedFormExpectations
    using Distributions
    using ExponentialFamily
    using StableRNGs
    using LinearAlgebra

    # Simple case: Normal to Normal
    target_dist = Normal(5.0, 2.0)
    target = Logpdf(target_dist)

    initial = Normal(0.0, 1.0)

    # Analytic
    t_analytic = @elapsed begin
        res_analytic = project_to(
            ProjectedTo(NormalMeanVariance),
            target;
            strategy = ClosedFormStrategy(),
            initial_point = initial,
            parameters = ProjectionParameters(niterations = 100),
        )
    end

    # MC
    t_mc = @elapsed begin
        res_mc = project_to(
            ProjectedTo(NormalMeanVariance),
            target;
            strategy = ControlVariateStrategy(nsamples = 1000),
            initial_point = initial,
            parameters = ProjectionParameters(niterations = 100),
        )
    end

    println("ClosedFormStrategy time: $t_analytic")
    println("ControlVariateStrategy time: $t_mc")

    # Analytic should be more accurate (converge to exact target)
    @test isapprox(mean(res_analytic), mean(target_dist), atol = 2e-2)
    @test isapprox(std(res_analytic), std(target_dist), atol = 2e-2)

    # ClosedFormStrategy should be at least as accurate as MC
    @test abs(mean(res_analytic) - 5.0) <= abs(mean(res_mc) - 5.0) + 0.1
end



@testitem "ClosedFormStrategy ProjectionCostGradientObjective integration" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using ExponentialFamily
    using Distributions
    using ExponentialFamilyManifolds
    using Manifolds
    using StableRNGs
    import ExponentialFamilyProjection: ProjectionCostGradientObjective, create_state!

    # Test that the objective works correctly with ClosedFormStrategy
    strategy = ClosedFormStrategy()
    target_dist = Normal(2.0, 1.0)
    target = Logpdf(target_dist)

    q_dist = NormalMeanVariance(0.0, 1.0)
    ef = convert(ExponentialFamilyDistribution, q_dist)

    M = ExponentialFamilyManifolds.get_natural_manifold(NormalMeanVariance, ())
    η = getnaturalparameters(ef)

    parameters = ProjectionParameters(rng = StableRNG(42))
    state = create_state!(strategy, M, parameters, target, ef, ())

    obj = ProjectionCostGradientObjective(parameters, target, copy(η), (), strategy, state)

    # Test evaluation
    p = ExponentialFamilyManifolds.partition_point(M, η)
    X = Manifolds.zero_vector(M, p)

    cost, X_result = obj(M, X, p)

    @test isfinite(cost)
    @test cost > 0  # KL divergence should be positive
    @test all(isfinite.(X_result))
end

@testitem "ClosedFormStrategy logbasemeasure_correction for ConstantBaseMeasure" begin
    using ExponentialFamilyProjection
    using ClosedFormExpectations
    using ExponentialFamily
    using Distributions
    using Test

    # Get the extension module
    # The extension should be loaded because both ExponentialFamilyProjection and ClosedFormExpectations are loaded
    ClosedFormExpectationsExt = Base.get_extension(ExponentialFamilyProjection, :ClosedFormExpectationsExt)
    
    @test !isnothing(ClosedFormExpectationsExt)

    strategy = ClosedFormStrategy()
    
    # Create a mock or usage that triggers ConstantBaseMeasure
    # We will use reflection/internals to test the specific method
    
    # Using a distribution that we know has ConstantBaseMeasure
    # or constructing it manually if possible.
    # ExponentialFamily.ConstantBaseMeasure is a singleton struct usually.
    
    base_measure = ExponentialFamily.ConstantBaseMeasure()
    q_dist = Normal(0, 1) # Any distribution works as q_dist is passed through
    grad_target = [1.0, 2.0]
    
    result = ClosedFormExpectationsExt.logbasemeasure_correction(
        strategy,
        base_measure,
        q_dist,
        grad_target
    )
    
    # The function should return grad_target exactly for ConstantBaseMeasure
    @test result === grad_target
end
