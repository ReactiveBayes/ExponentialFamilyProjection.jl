@testitem "InplaceLogpdfGradHess construction and basic functionality" begin
    import ExponentialFamilyProjection: InplaceLogpdfGradHess
    using LinearAlgebra

    # Define simple functions for testing
    logpdf_fn! = (out, x) -> out .= -(x .- 1).^2
    grad_fn! = (out, x) -> out .= -2 .* (x .- 1)
    hess_fn! = (out, x) -> out .= -2 .* I

    # Test construction
    inplace_struct = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)
    
    @test inplace_struct isa InplaceLogpdfGradHess
    @test inplace_struct.logpdf! === logpdf_fn!
    @test inplace_struct.grad! === grad_fn!
    @test inplace_struct.hess! === hess_fn!
end

@testitem "InplaceLogpdfGradHess univariate case" begin
    import ExponentialFamilyProjection: InplaceLogpdfGradHess
    
    # Univariate quadratic: -(x-1)²
    # logpdf: -(x-1)²
    # grad: -2(x-1)
    # hess: -2
    
    logpdf_fn! = (out, x) -> (out[1] = -(x - 1)^2)
    grad_fn! = (out, x) -> (out[1] = -2 * (x - 1))
    hess_fn! = (out, x) -> (out[1] = -2)
    
    inplace_struct = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)
    
    # Test points
    test_points = [0.0, 1.0, 2.0, 3.0]
    
    for x in test_points
        # Test logpdf
        logpdf_out = zeros(1)
        ExponentialFamilyProjection.logpdf!(inplace_struct, logpdf_out, x)
        expected_logpdf = -(x - 1)^2
        @test logpdf_out[1] ≈ expected_logpdf
        
        # Test gradient
        grad_out = zeros(1)
        ExponentialFamilyProjection.grad!(inplace_struct, grad_out, x)
        expected_grad = -2 * (x - 1)
        @test grad_out[1] ≈ expected_grad
        
        # Test hessian
        hess_out = zeros(1)
        ExponentialFamilyProjection.hess!(inplace_struct, hess_out, x)
        expected_hess = -2
        @test hess_out[1] ≈ expected_hess
    end
end

@testitem "InplaceLogpdfGradHess multivariate case" begin
    import ExponentialFamilyProjection: InplaceLogpdfGradHess
    using LinearAlgebra
    
    # Multivariate quadratic: -(x₁-1)² - (x₂-2)²
    # logpdf: -(x₁-1)² - (x₂-2)²
    # grad: [-2(x₁-1), -2(x₂-2)]
    # hess: [[-2, 0], [0, -2]]
    
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
    
    inplace_struct = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)
    
    # Test points
    test_points = [
        [0.0, 0.0],
        [1.0, 2.0],  # optimal point
        [2.0, 3.0],
        [0.5, 1.5]
    ]
    
    for x in test_points
        # Test logpdf
        logpdf_out = zeros(1)
        ExponentialFamilyProjection.logpdf!(inplace_struct, logpdf_out, x)
        expected_logpdf = -(x[1] - 1)^2 - (x[2] - 2)^2
        @test logpdf_out[1] ≈ expected_logpdf
        
        # Test gradient
        grad_out = zeros(2)
        ExponentialFamilyProjection.grad!(inplace_struct, grad_out, x)
        expected_grad = [-2 * (x[1] - 1), -2 * (x[2] - 2)]
        @test grad_out ≈ expected_grad
        
        # Test hessian
        hess_out = zeros(2, 2)
        ExponentialFamilyProjection.hess!(inplace_struct, hess_out, x)
        expected_hess = [-2 0; 0 -2]
        @test hess_out ≈ expected_hess
    end
end

@testitem "InplaceLogpdfGradHess higher dimensional case" begin
    import ExponentialFamilyProjection: InplaceLogpdfGradHess
    using LinearAlgebra
    
    # 3D case: -(x₁-1)² - (x₂-2)² - (x₃-3)²
    dim = 3
    targets = [1.0, 2.0, 3.0]
    
    logpdf_fn! = (out, x) -> begin
        out[1] = -sum((x[i] - targets[i])^2 for i in 1:dim)
    end
    grad_fn! = (out, x) -> begin
        for i in 1:dim
            out[i] = -2 * (x[i] - targets[i])
        end
    end
    hess_fn! = (out, x) -> begin
        fill!(out, 0)
        for i in 1:dim
            out[i, i] = -2
        end
    end
    
    inplace_struct = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)
    
    # Test points
    test_points = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],  # optimal point
        [2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5]
    ]
    
    for x in test_points
        # Test logpdf
        logpdf_out = zeros(1)
        ExponentialFamilyProjection.logpdf!(inplace_struct, logpdf_out, x)
        expected_logpdf = -sum((x[i] - targets[i])^2 for i in 1:dim)
        @test logpdf_out[1] ≈ expected_logpdf
        
        # Test gradient
        grad_out = zeros(dim)
        ExponentialFamilyProjection.grad!(inplace_struct, grad_out, x)
        expected_grad = [-2 * (x[i] - targets[i]) for i in 1:dim]
        @test grad_out ≈ expected_grad
        
        # Test hessian
        hess_out = zeros(dim, dim)
        ExponentialFamilyProjection.hess!(inplace_struct, hess_out, x)
        expected_hess = -2 * I(dim)
        @test hess_out ≈ expected_hess
    end
end

@testitem "InplaceLogpdfGradHess edge cases and validation" begin
    import ExponentialFamilyProjection: InplaceLogpdfGradHess
    
    # Test with different container sizes
    logpdf_fn! = (out, x) -> (out[1] = -(x[1] - 1)^2)
    grad_fn! = (out, x) -> (out[1] = -2 * (x[1] - 1))
    hess_fn! = (out, x) -> (out[1, 1] = -2)
    
    inplace_struct = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)
    
    x = [2.0]
    
    # Test that functions modify the containers correctly
    logpdf_out = ones(1)  # start with non-zero values
    ExponentialFamilyProjection.logpdf!(inplace_struct, logpdf_out, x)
    @test logpdf_out[1] ≈ -1.0  # -(2-1)² = -1
    
    grad_out = ones(1)
    ExponentialFamilyProjection.grad!(inplace_struct, grad_out, x)
    @test grad_out[1] ≈ -2.0  # -2(2-1) = -2
    
    hess_out = ones(1, 1)
    ExponentialFamilyProjection.hess!(inplace_struct, hess_out, x)
    @test hess_out[1, 1] ≈ -2.0
end

@testitem "BonnetStrategy vs ControlVariateStrategy comparison" begin
    using ExponentialFamily,
        Distributions,
        BayesBase,
        LinearAlgebra,
        Random,
        StableRNGs,
        ExponentialFamilyManifolds
    import ExponentialFamilyProjection:
        BonnetStrategy,
        BonnetStrategyState,
        ControlVariateStrategy,
        ControlVariateStrategyState,
        InplaceLogpdfGradHess,
        prepare_state!,
        get_samples,
        get_logpdfs,
        get_grads,
        get_hessians,
        get_current_mean,
        get_logbasemeasures,
        get_sufficientstatistics,
        get_gradsamples,
        ProjectionParameters,
        compute_gradient!

    # Test with multivariate normal distribution
    μ = [1.0, 2.0]
    Σ = [2.0 0.5; 0.5 1.0]
    dist = MvNormalMeanCovariance(μ, Σ)
    
    # Create specific target function: -(x₁-1)² - (x₂-2)²
    target_logpdf = (x) -> -(x[1] - 1)^2 - (x[2] - 2)^2
    
    # Create InplaceLogpdfGradHess for BonnetStrategy
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
    bonnet_target = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)
        
    # Test parameters
    nsamples = 80000
    seed = 42
    sample_dim = 2
    
    # Create exponential family distribution and manifold
    ef = convert(ExponentialFamilyDistribution, dist)
    T = ExponentialFamily.exponential_family_typetag(ef)
    d = size(mean(ef))
    c = getconditioner(ef)
    M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)
    
    # Use the same initial point for both strategies
    rng = StableRNG(seed)
    initial_point = rand(rng, M)
    initial_ef = convert(ExponentialFamilyDistribution, M, initial_point)
    
    # Create strategies with same nsamples
    bonnet_strategy = BonnetStrategy(nsamples = nsamples)
    control_variate_strategy = ControlVariateStrategy(nsamples = nsamples, buffer = nothing)
    
    # Test with the same seed for reproducibility
    bonnet_parameters = ProjectionParameters(rng = StableRNG(seed), seed = seed)
    cv_parameters = ProjectionParameters(rng = StableRNG(seed), seed = seed)
    
    # Preprocess the strategy arguments to handle conversion properly
    import ExponentialFamilyProjection: preprocess_strategy_argument
    bonnet_strategy_processed, bonnet_target_processed = preprocess_strategy_argument(bonnet_strategy, bonnet_target)
    cv_strategy_processed, cv_target_processed = preprocess_strategy_argument(control_variate_strategy, target_logpdf)
    
    # Create containers for BonnetStrategy
    bonnet_samples = rand(initial_ef, nsamples)
    bonnet_logpdfs = zeros(nsamples)
    bonnet_grads = zeros(sample_dim, nsamples)
    bonnet_hessians = zeros(sample_dim, sample_dim, nsamples)
    bonnet_current_mean = zeros(sample_dim)
    
    bonnet_state = BonnetStrategyState(
        samples = bonnet_samples,
        logpdfs = bonnet_logpdfs,
        grads = bonnet_grads,
        hessians = bonnet_hessians,
        current_mean = bonnet_current_mean
    )
    
    # Create containers for ControlVariateStrategy  
    cv_samples = rand(initial_ef, nsamples)
    cv_logpdfs = zeros(nsamples)
    cv_logbasemeasures = zeros(nsamples)  # We'll handle base measures
    cv_sufficientstatistics = zeros(length(getnaturalparameters(initial_ef)), nsamples)
    cv_gradsamples = zeros(length(getnaturalparameters(initial_ef)), nsamples)
    
    cv_state = ControlVariateStrategyState(
        samples = cv_samples,
        logpdfs = cv_logpdfs,
        logbasemeasures = cv_logbasemeasures,
        sufficientstatistics = cv_sufficientstatistics,
        gradsamples = cv_gradsamples
    )
    
    # Prepare states
    supplementary_η = ()
    bonnet_state_prepared = prepare_state!(
        bonnet_strategy_processed,
        bonnet_state,
        M,
        bonnet_parameters,
        bonnet_target_processed,
        initial_ef,
        supplementary_η
    )
    
    cv_state_prepared = prepare_state!(
        cv_strategy_processed,
        cv_state,
        M,
        cv_parameters,
        cv_target_processed,
        initial_ef,
        supplementary_η
    )
    
    # Verify both strategies are using the same samples (they should be with same seed)
    @test get_samples(bonnet_state_prepared) ≈ get_samples(cv_state_prepared)
    
    # Get some parameters from the initial distribution
    η = getnaturalparameters(initial_ef)
    logpartition = ExponentialFamily.logpartition(initial_ef)
    gradlogpartition = ExponentialFamily.gradlogpartition(initial_ef)
    fisherinformation = ExponentialFamily.fisherinformation(initial_ef)
    inv_fisher = inv(fisherinformation)
    
    # Create gradient containers
    bonnet_gradient = zeros(length(η))
    cv_gradient = zeros(length(η))
    
    # Compute gradients using both strategies
    compute_gradient!(
        M,
        bonnet_strategy_processed,
        bonnet_state_prepared,
        bonnet_gradient,
        η
    )
    
    compute_gradient!(
        M,
        cv_strategy_processed,
        cv_state_prepared,
        cv_gradient,
        η,
        logpartition,
        gradlogpartition,
        inv_fisher
    )
    
    # Compare the gradients - they should be approximately equal
    grad_diff = bonnet_gradient - cv_gradient
    @test dot(grad_diff, fisherinformation, grad_diff) ≈ 0 atol=1e-3
    
    # Additional verification: test that both strategies produce finite results
    @test all(isfinite, bonnet_gradient)
    @test all(isfinite, cv_gradient)
    
    # Test with different initial points to ensure consistency
    for test_seed in [123, 456, 789]
        test_rng = StableRNG(test_seed)
        test_point = rand(test_rng, M)
        test_ef = convert(ExponentialFamilyDistribution, M, test_point)
        
        bonnet_params_test = ProjectionParameters(rng = StableRNG(test_seed), seed = test_seed)
        cv_params_test = ProjectionParameters(rng = StableRNG(test_seed), seed = test_seed)
        
        # Prepare states with new test point
        prepare_state!(
            bonnet_strategy_processed,
            bonnet_state,
            M,
            bonnet_params_test,
            bonnet_target_processed,
            test_ef,
            supplementary_η
        )
        
        prepare_state!(
            cv_strategy_processed,
            cv_state,
            M,
            cv_params_test,
            cv_target_processed,
            test_ef,
            supplementary_η
        )
        
        # Get parameters for test point
        test_η = getnaturalparameters(test_ef)
        test_logpartition = ExponentialFamily.logpartition(test_ef)
        test_gradlogpartition = ExponentialFamily.gradlogpartition(test_ef)
        test_inv_fisher = inv(ExponentialFamily.fisherinformation(test_ef))
        
        # Reset gradient containers
        fill!(bonnet_gradient, 0.0)
        fill!(cv_gradient, 0.0)
        
        # Compute gradients
        compute_gradient!(
            M,
            bonnet_strategy_processed,
            bonnet_state,
            bonnet_gradient,
            test_η,
        )
        
        compute_gradient!(
            M,
            cv_strategy_processed,
            cv_state,
            cv_gradient,
            test_η,
            test_logpartition,
            test_gradlogpartition,
            test_inv_fisher
        )
        
        # Compare gradients for this test point
        grad_diff = bonnet_gradient - cv_gradient
        @test dot(grad_diff, fisherinformation, grad_diff) ≈ 0 atol=1e-3
    end
end

@testitem "BonnetStrategy getter functions" begin
    import ExponentialFamilyProjection:
        BonnetStrategy,
        BonnetStrategyState,
        get_nsamples,
        get_samples,
        get_logpdfs,
        get_grads,
        get_hessians,
        get_current_mean
    using LinearAlgebra

    # Test BonnetStrategy getter
    strategy = BonnetStrategy(nsamples = 500)
    @test get_nsamples(strategy) === 500

    strategy = BonnetStrategy(nsamples = 1000)
    @test get_nsamples(strategy) === 1000

    # Test BonnetStrategyState getters
    nsamples = 100
    sample_dim = 3
    
    samples = randn(sample_dim, nsamples)
    logpdfs = zeros(nsamples)
    grads = randn(sample_dim, nsamples)
    hessians = randn(sample_dim, sample_dim, nsamples)
    current_mean = randn(sample_dim)
    
    state = BonnetStrategyState(
        samples = samples,
        logpdfs = logpdfs,
        grads = grads,
        hessians = hessians,
        current_mean = current_mean
    )
    
    @test get_samples(state) === samples
    @test get_logpdfs(state) === logpdfs
    @test get_grads(state) === grads
    @test get_hessians(state) === hessians
    @test get_current_mean(state) === current_mean
end

@testitem "BonnetStrategy prepare_state! for multivariate normal" begin
    using ExponentialFamily,
        Distributions,
        BayesBase,
        LinearAlgebra,
        Random,
        StableRNGs,
        ExponentialFamilyManifolds
    import ExponentialFamilyProjection:
        BonnetStrategy,
        BonnetStrategyState,
        InplaceLogpdfGradHess,
        prepare_state!,
        get_samples,
        get_logpdfs,
        get_grads,
        get_hessians,
        get_current_mean,
        ProjectionParameters

    # Test with multivariate normal distribution
    μ = [1.0, 2.0]
    Σ = [2.0 0.5; 0.5 1.0]
    dist = MvNormalMeanCovariance(μ, Σ)
    
    # Create target function: multivariate quadratic -(x₁-1)² - (x₂-2)²
    targetfn = (x) -> -(x[1] - 1)^2 - (x[2] - 2)^2
    
    # Create InplaceLogpdfGradHess manually
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
    sample_dim = 2
    
    # Pre-create containers
    samples = rand(dist, nsamples)  # This creates (2, nsamples) matrix
    logpdfs = zeros(nsamples)
    grads = zeros(sample_dim, nsamples)  # gradient at each sample
    hessians = zeros(sample_dim, sample_dim, nsamples)  # hessian at each sample
    current_mean = zeros(sample_dim)
    
    state = BonnetStrategyState(
        samples = samples,
        logpdfs = logpdfs,
        grads = grads,
        hessians = hessians,
        current_mean = current_mean
    )
    
    # Create exponential family distribution and parameters
    ef = convert(ExponentialFamilyDistribution, dist)
    T = ExponentialFamily.exponential_family_typetag(ef)
    d = size(mean(ef))
    c = getconditioner(ef)
    M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)
    
    strategy = BonnetStrategy(nsamples = nsamples)
    rng = StableRNG(42)
    parameters = ProjectionParameters(rng = rng)
    
    # Test prepare_state! (ignoring the current_mean computation for now due to dim_size issue)
    prepare_state!(strategy, state, M, parameters, inplace_target, ef, getnaturalparameters(ef))
    
    # Test that the containers are filled correctly
    @test all(isfinite, get_logpdfs(state))
    @test all(isfinite, get_grads(state))
    @test all(isfinite, get_hessians(state))

    _, sample_container = ExponentialFamily.check_logpdf(ef, get_samples(state))

    # Manually evaluate logpdf, grad, hess for each sample to verify
    for (i, sample) in enumerate(sample_container)
        # Test logpdf evaluation
        logpdf_out = zeros(1)
        ExponentialFamilyProjection.logpdf!(inplace_target, logpdf_out, sample)
        @test logpdf_out[1] ≈ get_logpdfs(state)[i]
        
        # Test gradient evaluation  
        grad_out = zeros(2)
        ExponentialFamilyProjection.grad!(inplace_target, grad_out, sample)
        @test grad_out ≈ get_grads(state)[:, i]
        
        # Test hessian evaluation
        hess_out = zeros(2, 2)
        ExponentialFamilyProjection.hess!(inplace_target, hess_out, sample)
        @test hess_out ≈ get_hessians(state)[:, :, i]
    end

    @test get_current_mean(state) ≈ mean(dist)
end

@testitem "BonnetStrategy prepare_state! for univariate normal" begin
    using ExponentialFamily,
        Distributions,
        BayesBase,
        LinearAlgebra,
        Random,
        StableRNGs,
        ExponentialFamilyManifolds
    import ExponentialFamilyProjection:
        BonnetStrategy,
        BonnetStrategyState,
        InplaceLogpdfGradHess,
        prepare_state!,
        get_samples,
        get_logpdfs,
        get_grads,
        get_hessians,
        get_current_mean,
        ProjectionParameters

    # Test with univariate normal distribution
    dist = NormalMeanVariance(1.0, 2.0)
    
    # Create target function: univariate quadratic -(x-1)²
    targetfn = (x) -> -(x - 1)^2
    
    # Create InplaceLogpdfGradHess manually
    logpdf_fn! = (out, x) -> (out[1] = -(x - 1)^2)
    grad_fn! = (out, x) -> (out[1] = -2 * (x - 1))
    hess_fn! = (out, x) -> (out[1] = -2)
    inplace_target = InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)
    
    nsamples = 30
    sample_dim = 1
    
    # Pre-create containers for univariate case
    samples = rand(dist, nsamples)  # This creates a vector of length nsamples
    logpdfs = zeros(nsamples)
    grads = zeros(sample_dim, nsamples)  # (1, nsamples) matrix
    hessians = zeros(sample_dim, sample_dim, nsamples)  # (1, 1, nsamples) array
    current_mean = zeros(sample_dim)
    
    state = BonnetStrategyState(
        samples = samples,
        logpdfs = logpdfs,
        grads = grads,
        hessians = hessians,
        current_mean = current_mean
    )
    
    # Create exponential family distribution and parameters
    ef = convert(ExponentialFamilyDistribution, dist)
    T = ExponentialFamily.exponential_family_typetag(ef)
    d = size(mean(ef))
    c = getconditioner(ef)
    M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)
    
    strategy = BonnetStrategy(nsamples = nsamples)
    rng = StableRNG(42)
    parameters = ProjectionParameters(rng = rng)

    prepare_state!(strategy, state, M, parameters, inplace_target, ef, getnaturalparameters(ef))
    
    # Verify containers are filled correctly
    @test all(isfinite, get_logpdfs(state))
    @test all(isfinite, get_grads(state))
    @test all(isfinite, get_hessians(state))
    
    _, sample_container = ExponentialFamily.check_logpdf(ef, get_samples(state))
    
    # Manually evaluate for each sample
    for (i, sample) in enumerate(sample_container)
        # Test logpdf evaluation
        logpdf_out = zeros(1)
        ExponentialFamilyProjection.logpdf!(inplace_target, logpdf_out, sample)
        @test logpdf_out[1] ≈ get_logpdfs(state)[i]
        
        # Test gradient evaluation  
        grad_out = zeros(1)
        ExponentialFamilyProjection.grad!(inplace_target, grad_out, sample)
        @test grad_out[1] ≈ get_grads(state)[1, i]
        
        # Test hessian evaluation
        hess_out = zeros(1, 1)
        ExponentialFamilyProjection.hess!(inplace_target, hess_out, sample)
        @test hess_out[1, 1] ≈ get_hessians(state)[1, 1, i]
    end
    
end

@testitem "`BonnetStrategy` should fail if given a list of samples instead of a function" begin
    using ExponentialFamily

    prj = ProjectedTo(
        NormalMeanVariance;
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.BonnetStrategy(),
        ),
    )

    @test_throws "The `BonnetStrategy` requires the projection argument to be a callable object (e.g. `Function`) or an `InplaceLogpdfGradHess`. Got `Vector{Float64}` instead." project_to(
        prj,
        [0.5],
    )
end

@testitem "`BonnetStrategy` target function should compute gradients correctly for Normal distributions" begin
    using ExponentialFamily,
        LinearAlgebra,
        Distributions,
        ExponentialFamilyManifolds,
        ExponentialFamilyProjection,
        StableRNGs,
        ForwardDiff,
        BayesBase,
        Manifolds

    rng = StableRNG(42)
    for distribution in [
            NormalMeanVariance(0, 1),
            MvNormalMeanCovariance(ones(2), Matrix(Diagonal(ones(2)))),
        ]

        # Create target distribution to project to
        target_dist = distribution
        targetfn = (x) -> logpdf(target_dist, x)

        # Create InplaceLogpdfGradHess for BonnetStrategy
        if distribution isa NormalMeanVariance
            # Univariate case
            logpdf_fn! = (out, x) -> (out[1] = logpdf(target_dist, x))
            grad_fn! = (out, x) -> (out[1] = ForwardDiff.derivative(x -> logpdf(target_dist, x), x))
            hess_fn! = (out, x) -> (out[1] = ForwardDiff.derivative(x -> ForwardDiff.derivative(x -> logpdf(target_dist, x), x), x))
        else
            # Multivariate case
            logpdf_fn! = (out, x) -> (out[1] = logpdf(target_dist, x))
            grad_fn! = (out, x) -> (out .= ForwardDiff.gradient(x -> logpdf(target_dist, x), x))
            hess_fn! = (out, x) -> (out .= ForwardDiff.hessian(x -> logpdf(target_dist, x), x))
        end
        
        bonnet_target = ExponentialFamilyProjection.InplaceLogpdfGradHess(logpdf_fn!, grad_fn!, hess_fn!)

        ef = convert(ExponentialFamilyDistribution, distribution)
        T = ExponentialFamily.exponential_family_typetag(ef)
        c = getconditioner(ef)
        d = size(rand(rng, ef))
        M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)
        
        # Test with both BonnetStrategy and ControlVariateStrategy for comparison
        bonnet_strategy = ExponentialFamilyProjection.BonnetStrategy(nsamples = 10000)
        cv_strategy = ExponentialFamilyProjection.ControlVariateStrategy(nsamples = 10000)
        
        p = ProjectionParameters(rng = StableRNG(42), seed = 42)
        η = getnaturalparameters(ef)

        # Test BonnetStrategy
        bonnet_state = ExponentialFamilyProjection.create_state!(bonnet_strategy, M, p, bonnet_target, ef, ())
        bonnet_obj = ExponentialFamilyProjection.ProjectionCostGradientObjective(
            p,
            bonnet_target,
            copy(η),
            (),
            bonnet_strategy,
            bonnet_state,
        )

        # Test ControlVariateStrategy for comparison
        cv_state = ExponentialFamilyProjection.create_state!(cv_strategy, M, p, targetfn, ef, ())
        cv_obj = ExponentialFamilyProjection.ProjectionCostGradientObjective(
            p,
            targetfn,
            copy(η),
            (),
            cv_strategy,
            cv_state,
        )

        @test bonnet_state == bonnet_state
        @test cv_state == cv_state

        _logpartition = logpartition(ef)
        _gradlogpartition = gradlogpartition(ef)
        _inv_fisher = inv(fisherinformation(ef))
        
        # Compute costs
        bonnet_cost = ExponentialFamilyProjection.compute_cost(
            M,
            bonnet_strategy,
            bonnet_state,
            η,
            _logpartition,
            _gradlogpartition,
            _inv_fisher,
        )

        cv_cost = ExponentialFamilyProjection.compute_cost(
            M,
            cv_strategy,
            cv_state,
            η,
            _logpartition,
            _gradlogpartition,
            _inv_fisher,
        )

        # Compute gradients
        bonnet_gradient = similar(η)
        cv_gradient = similar(η)

        ExponentialFamilyProjection.compute_gradient!(
            M,
            bonnet_strategy,
            bonnet_state,
            bonnet_gradient,
            η,
            _logpartition,
            _gradlogpartition,
            _inv_fisher,
        )

        ExponentialFamilyProjection.compute_gradient!(
            M,
            cv_strategy,
            cv_state,
            cv_gradient,
            η,
            _logpartition,
            _gradlogpartition,
            _inv_fisher,
        )

        # The costs should be approximately equal (both targeting the same distribution)
        @test bonnet_cost ≈ cv_cost rtol = 1e-2

        # The gradients should be approximately equal
        @test bonnet_gradient ≈ cv_gradient rtol = 1e-2

        # Test gradient computation in manifold coordinates
        p_manifold = ExponentialFamilyManifolds.partition_point(M, η)
        
        X_p_bonnet = Manifolds.zero_vector(M, p_manifold)
        X_p_cv = Manifolds.zero_vector(M, p_manifold)
        
        c_p_bonnet, X_p_bonnet = bonnet_obj(M, X_p_bonnet, p_manifold)
        c_p_cv, X_p_cv = cv_obj(M, X_p_cv, p_manifold)
        
        @test c_p_bonnet ≈ bonnet_cost
        @test c_p_cv ≈ cv_cost
        @test X_p_bonnet ≈ X_p_cv rtol = 1e-2
    end
end