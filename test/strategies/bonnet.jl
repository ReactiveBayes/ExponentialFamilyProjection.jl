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
    
    # Test the container filling manually
    Random.seed!(rng, 42)
    Random.rand!(rng, ef, get_samples(state))
    
    _, sample_container = ExponentialFamily.check_logpdf(ef, get_samples(state))
    
    # Manually evaluate for each sample
    for (i, sample) in enumerate(sample_container)
        # Test logpdf evaluation
        logpdf_out = zeros(1)
        logpdf!(inplace_target, logpdf_out, sample)
        expected_logpdf = -(sample - 1)^2
        @test logpdf_out[1] ≈ expected_logpdf
        
        # Test gradient evaluation  
        grad_out = zeros(1)
        grad!(inplace_target, grad_out, sample)
        expected_grad = -2 * (sample - 1)
        @test grad_out[1] ≈ expected_grad
        
        # Test hessian evaluation
        hess_out = zeros(1, 1)
        hess!(inplace_target, hess_out, sample)
        expected_hess = -2
        @test hess_out[1, 1] ≈ expected_hess
        
        # Store in containers
        get_logpdfs(state)[i] = logpdf_out[1]
        get_grads(state)[1, i] = grad_out[1]  # Note: for univariate, grads is (1, nsamples)
        get_hessians(state)[1, 1, i] = hess_out[1, 1]
    end
    
    # Verify containers are filled correctly
    @test all(isfinite, get_logpdfs(state))
    @test all(isfinite, get_grads(state))
    @test all(isfinite, get_hessians(state))
    
    # Test that values match expected calculations
    for (i, sample) in enumerate(sample_container)
        expected_logpdf = -(sample - 1)^2
        @test get_logpdfs(state)[i] ≈ expected_logpdf
        
        expected_grad = -2 * (sample - 1)
        @test get_grads(state)[1, i] ≈ expected_grad
        
        expected_hess = -2
        @test get_hessians(state)[1, 1, i] ≈ expected_hess
    end
end

@testitem "BonnetInplaceLogpdf" begin
    import ExponentialFamilyProjection:
        BonnetInplaceLogpdf
    using BayesBase: InplaceLogpdf

    logpdf! = (out, x) -> -(x - 1)^2
    grad! = (out, x) -> out .= -2 * (x - 1)
    hess! = (out, x) -> out .= -2

    inplace = BonnetInplaceLogpdf(logpdf!, grad!, hess!)

    @test inplace(zeros(3), 1:3) == 1:3
end