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