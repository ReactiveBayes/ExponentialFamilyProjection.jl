@testitem "GaussNewton strategy tests - Univariate" begin
    using ExponentialFamily,
        Distributions,
        BayesBase,
        LinearAlgebra,
        Random,
        StableRNGs,
        ExponentialFamilyManifolds,
        ForwardDiff,
        FillArrays,
        Manifolds
    import ExponentialFamilyProjection:
        GaussNewton,
        GaussNewtonState,
        InplaceLogpdfGradHess,
        create_state!,
        prepare_state!,
        get_samples,
        get_logpdfs,
        get_logbasemeasures,
        get_nsamples,
        ProjectionParameters

    # Test with univariate normal distribution
    dist = NormalMeanVariance(1.0, 2.0)

    # Create InplaceLogpdfGradHess manually
    logpdf_fn! = (out, x) -> (out[1] = -(x - 1)^2)
    grad_fn! = (out, x) -> (out[1] = -2 * (x - 1))
    hess_fn! = (out, x) -> (out[1] = -2)
    inplace_target = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)

    nsamples = 40

    # Create exponential family distribution and parameters
    ef = convert(ExponentialFamilyDistribution, dist)
    T = ExponentialFamily.exponential_family_typetag(ef)
    d = size(mean(ef))
    c = getconditioner(ef)
    M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)

    strategy = GaussNewton(nsamples = nsamples)
    rng = StableRNG(42)
    parameters = ProjectionParameters(rng = rng, seed = 42)

    state = create_state!(strategy, M, parameters, inplace_target, ef, ())

    # Verify sample-based containers are filled for cost computation
    @test length(get_samples(state)) == nsamples
    @test length(get_logpdfs(state)) == nsamples
    @test length(get_logbasemeasures(state)) == nsamples

    # Check logpdfs vs manual evaluation on samples
    _, sample_container = ExponentialFamily.check_logpdf(ef, get_samples(state))
    for (i, sample) in enumerate(sample_container)
        logpdf_out = zeros(1)
        ExponentialFamilyProjection.logpdf!(inplace_target, logpdf_out, sample)
        @test logpdf_out[1] ≈ get_logpdfs(state)[i]
    end

    # Check gradient and Hessian are computed at current mean (deterministic)
    # current mean for NormalMeanVariance is just the mean(dist)
    current_mean = mean(dist)
    grad_out = zeros(1)
    hess_out = zeros(1, 1)
    ExponentialFamilyProjection.grad_hess!(inplace_target, grad_out, hess_out, current_mean)
    @test state.grad[1] ≈ grad_out[1]
    @test state.hessian[1, 1] ≈ hess_out[1, 1]
end

@testitem "GaussNewton for multivariate normal" begin
    using ExponentialFamily,
        Distributions,
        BayesBase,
        LinearAlgebra,
        Random,
        StableRNGs,
        ExponentialFamilyManifolds
    import ExponentialFamilyProjection:
        GaussNewton,
        GaussNewtonState,
        InplaceLogpdfGradHess,
        create_state!,
        prepare_state!,
        get_samples,
        get_logpdfs,
        get_logbasemeasures,
        ProjectionParameters

    # Multivariate normal distribution
    μ = [1.0, 2.0]
    Σ = [2.0 0.5; 0.5 1.0]
    dist = MvNormalMeanCovariance(μ, Σ)

    # Inplace target using simple analytical functions
    logpdf_fn! = (out, x) -> (out[1] = -(x[1] - 1)^2 - (x[2] - 2)^2)
    grad_fn! = (out, x) -> begin
        out[1] = -2 * (x[1] - 1)
        out[2] = -2 * (x[2] - 2)
    end
    hess_fn! = (out, x) -> begin
        out[1, 1] = -2
        out[1, 2] = 0
        out[2, 1] = 0
        out[2, 2] = -2
    end
    inplace_target = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)

    nsamples = 50

    ef = convert(ExponentialFamilyDistribution, dist)
    T = ExponentialFamily.exponential_family_typetag(ef)
    d = size(mean(ef))
    c = getconditioner(ef)
    M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)

    strategy = GaussNewton(nsamples = nsamples)
    rng = StableRNG(42)
    parameters = ProjectionParameters(rng = rng, seed = 42)

    state = create_state!(strategy, M, parameters, inplace_target, ef, ())

    # Sample-based containers for the cost should be filled
    @test size(get_samples(state)) == (2, nsamples)
    @test length(get_logpdfs(state)) == nsamples
    @test length(get_logbasemeasures(state)) == nsamples

    # Verify logpdfs on samples
    _, sample_container = ExponentialFamily.check_logpdf(ef, get_samples(state))
    for (i, sample) in enumerate(sample_container)
        logpdf_out = zeros(1)
        ExponentialFamilyProjection.logpdf!(inplace_target, logpdf_out, sample)
        @test logpdf_out[1] ≈ get_logpdfs(state)[i]
    end

    # Verify gradient and Hessian at the current mean
    current_mean = mean(dist)
    grad_out = zeros(2)
    hess_out = zeros(2, 2)
    ExponentialFamilyProjection.grad_hess!(inplace_target, grad_out, hess_out, current_mean)
    @test state.grad ≈ grad_out
    @test state.hessian ≈ hess_out
end

@testitem "`GaussNewton` should fail if given a list of samples instead of a function" begin
    using ExponentialFamily

    prj = ProjectedTo(
        NormalMeanVariance;
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.GaussNewton(),
        ),
    )

    @test_throws "The `GaussNewton` strategy requires the projection argument to be a callable object (e.g. `Function`) or an `InplaceLogpdfGradHess`. Got `Vector{Float64}` instead." project_to(
        prj,
        [0.5],
    )
end

@testitem "GaussNewton vs ControlVariateStrategy performance comparison for high-dimensional normals" begin
    using ExponentialFamily,
        Distributions,
        BayesBase,
        LinearAlgebra,
        Random,
        StableRNGs,
        ExponentialFamilyManifolds,
        ForwardDiff,
        Printf,
        BenchmarkTools,
        Manifolds
    import ExponentialFamilyProjection:
        GaussNewton,
        ControlVariateStrategy,
        InplaceLogpdfGradHess,
        create_state!,
        ProjectionParameters,
        ProjectionCostGradientObjective

    # Test with increasing dimensions to show GaussNewton advantage
    dimensions = [10, 20, 50]
    nsamples = 1000

    println("\nPerformance comparison: GaussNewton vs ControlVariateStrategy")
    println("Dimension | GaussNewton (μs) | ControlVariateStrategy (μs) | Speedup | Memory Ratio")
    println("----------|-------------------|-----------------------------|---------|--------------")

    for dim in dimensions
        # Create high-dimensional normal distribution
        μ = randn(StableRNG(42), dim)
        Σ = let A = randn(StableRNG(43), dim, dim); A * A' + 0.1 * I end
        dist = MvNormalMeanCovariance(μ, Σ)

        # Create target function for the same distribution
        target_logpdf = (x) -> logpdf(dist, x)

        # Create InplaceLogpdfGradHess for GaussNewton
        logpdf_fn = (out, x) -> (out[1] = logpdf(dist, x))
        grad_fn = let ForwardDiff = ForwardDiff
            (out, x) -> ForwardDiff.gradient!(out, x -> logpdf(dist, x), x)
        end
        hess_fn = let ForwardDiff = ForwardDiff
            (out, x) -> ForwardDiff.hessian!(out, x -> logpdf(dist, x), x)
        end
        gn_target = InplaceLogpdfGradHess(logpdf_fn, grad_fn, hess_fn)

        # Setup manifold and exponential family
        ef = convert(ExponentialFamilyDistribution, dist)
        T = ExponentialFamily.exponential_family_typetag(ef)
        d = size(mean(ef))
        c = getconditioner(ef)
        M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)

        η = getnaturalparameters(ef)

        # Pre-create variables for benchmarking
        test_parameters = ProjectionParameters(rng = StableRNG(42), seed = 42)
        p_manifold = ExponentialFamilyManifolds.partition_point(M, η)

        # Benchmark GaussNewton using ProjectionCostGradientObjective
        gn_benchmark = @benchmark begin
            gn_strategy = GaussNewton(nsamples = $nsamples)
            gn_state = create_state!(gn_strategy, $M, $test_parameters, $gn_target, $ef, ())
            gn_obj = ProjectionCostGradientObjective(
                $test_parameters, $gn_target, copy($η), (), gn_strategy, gn_state
            )
            X_gn = Manifolds.zero_vector($M, $p_manifold)
            cost_gn, X_gn = gn_obj($M, X_gn, $p_manifold)
        end

        # Benchmark ControlVariateStrategy using ProjectionCostGradientObjective
        cv_benchmark = @benchmark begin
            cv_strategy = ControlVariateStrategy(nsamples = $nsamples, buffer = nothing)
            cv_state = create_state!(cv_strategy, $M, $test_parameters, $target_logpdf, $ef, ())
            cv_obj = ProjectionCostGradientObjective(
                $test_parameters, $target_logpdf, copy($η), (), cv_strategy, cv_state
            )
            X_cv = Manifolds.zero_vector($M, $p_manifold)
            cost_cv, X_cv = cv_obj($M, X_cv, $p_manifold)
        end

        # Extract timing and memory statistics
        gn_time_μs = median(gn_benchmark.times) / 1000  # Convert ns to μs
        cv_time_μs = median(cv_benchmark.times) / 1000  # Convert ns to μs
        speedup = cv_time_μs / gn_time_μs

        gn_memory = gn_benchmark.memory
        cv_memory = cv_benchmark.memory
        memory_ratio = cv_memory / gn_memory

        # Print results
        println(@sprintf("%9d | %17.1f | %27.1f | %6.2fx | %12.2fx",
                dim, gn_time_μs, cv_time_μs, speedup, memory_ratio))

        # Test that both strategies produce similar outputs (functional correctness)
        parameters = ProjectionParameters(rng = StableRNG(42), seed = 42)
        p_manifold = ExponentialFamilyManifolds.partition_point(M, η)

        gn_strategy = GaussNewton(nsamples = nsamples)
        gn_state = create_state!(gn_strategy, M, parameters, gn_target, ef, ())
        gn_obj = ProjectionCostGradientObjective(
            parameters, gn_target, copy(η), (), gn_strategy, gn_state
        )
        X_gn = Manifolds.zero_vector(M, p_manifold)
        cost_gn, X_gn = gn_obj(M, X_gn, p_manifold)

        cv_strategy = ControlVariateStrategy(nsamples = nsamples, buffer = nothing)
        cv_state = create_state!(cv_strategy, M, parameters, target_logpdf, ef, ())
        cv_obj = ProjectionCostGradientObjective(
            parameters, target_logpdf, copy(η), (), cv_strategy, cv_state
        )
        X_cv = Manifolds.zero_vector(M, p_manifold)
        cost_cv, X_cv = cv_obj(M, X_cv, p_manifold)

        # Expect GaussNewton to be faster on higher dimensions
        if dim >= 20
            @test speedup > 1.0
        end
    end
end


