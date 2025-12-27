@testitem "DifferentiationInterface extension with GaussNewton strategy for ContinuousUni-/Multi-variateLogPdf" begin
    using ExponentialFamily, BayesBase
    using DifferentiationInterface
    using ADTypes
    import ExponentialFamilyProjection:
        GaussNewton,
        to_inplace_gradhess,
        ProjectionParameters,
        ProjectedTo,
        project_to
    import ExponentialFamily: NormalMeanVariance, MvNormalMeanCovariance
    import BayesBase: ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf
    import LinearAlgebra: Diagonal
    import DomainSets: ℝ

    # Test 1: Univariate case with default backend (AutoForwardDiff)
    @testset "Univariate with default AutoForwardDiff backend" begin
        a1 = NormalMeanVariance(-10.0, 0.1)
        my_logpdf(x) = logpdf(a1, x)
        my_uni_continuous_logpdf = ContinuousUnivariateLogPdf(my_logpdf)

        # Test explicit conversion
        default_uni_inplace = to_inplace_gradhess(my_uni_continuous_logpdf)
        @test default_uni_inplace isa InplaceLogpdfGradHess

        # Test projection with automatic conversion
        params = ProjectionParameters(niterations = 2000, strategy = GaussNewton())
        prj = ProjectedTo(NormalMeanVariance; parameters = params)

        test_uni_cont_proj = project_to(prj, my_uni_continuous_logpdf)
        test_uni_inplace_proj = project_to(prj, default_uni_inplace)

        @test test_uni_cont_proj ≈ test_uni_inplace_proj atol = 1e-6
        @test test_uni_cont_proj isa NormalMeanVariance
        @test mean(test_uni_cont_proj) ≈ mean(a1) atol = 1e-6
        @test var(test_uni_cont_proj) ≈ var(a1) atol = 1e-6
    end

    # Test 2: Multivariate case with default backend
    @testset "Multivariate with default AutoForwardDiff backend" begin
        a2 = MvNormalMeanCovariance([1.3, -5.0, 30.0], Diagonal([0.5, 2.0, 1.0]))
        my_logpdf(x) = logpdf(a2, x)
        my_mv_continuous_logpdf = ContinuousMultivariateLogPdf(ℝ^3, my_logpdf)

        # Test explicit conversion
        default_mv_inplace = to_inplace_gradhess(my_mv_continuous_logpdf)
        @test default_mv_inplace isa InplaceLogpdfGradHess

        # Test projection with automatic conversion
        params = ProjectionParameters(niterations = 2000, strategy = GaussNewton())
        prj = ProjectedTo(MvNormalMeanCovariance, 3; parameters = params)

        test_mv_cont_proj = project_to(prj, my_mv_continuous_logpdf)
        test_mv_inplace_proj = project_to(prj, default_mv_inplace)

        @test test_mv_cont_proj ≈ test_mv_inplace_proj atol = 1e-6
        @test test_mv_cont_proj isa MvNormalMeanCovariance
        @test mean(test_mv_cont_proj) ≈ mean(a2) atol = 1e-5
        @test cov(test_mv_cont_proj) ≈ cov(a2) atol = 1e-6
    end

    # Test 3: Custom backend specification
    @testset "Custom backend (AutoZygote)" begin
        using Zygote: Zygote

        a3 = MvNormalMeanCovariance([0.0, 0.0], Diagonal([1.0, 1.0]))
        my_logpdf(x) = logpdf(a3, x)
        my_mv_continuous_logpdf = ContinuousMultivariateLogPdf(ℝ^2, my_logpdf)

        # Test with AutoZygote backend
        zygote_inplace = to_inplace_gradhess(my_mv_continuous_logpdf, AutoZygote())
        @test zygote_inplace isa InplaceLogpdfGradHess

        params = ProjectionParameters(niterations = 2000, strategy = GaussNewton())
        prj = ProjectedTo(MvNormalMeanCovariance, 2; parameters = params)

        test_zygote_proj = project_to(prj, zygote_inplace)
        @test test_zygote_proj isa MvNormalMeanCovariance
        @test mean(test_zygote_proj) ≈ mean(a3) atol = 1e-5
        @test cov(test_zygote_proj) ≈ cov(a3) atol = 1e-5
    end
end

@testitem "DifferentiationInterface extension with BonnetStrategy for ContinuousMultivariateLogPdf" begin
    using ExponentialFamily, BayesBase
    using DifferentiationInterface
    using ADTypes
    import ExponentialFamilyProjection:
        BonnetStrategy,
        to_inplace_gradhess,
        ProjectionParameters,
        ProjectedTo,
        project_to
    import ExponentialFamily: MvNormalMeanCovariance
    import BayesBase: ContinuousMultivariateLogPdf
    import LinearAlgebra: Diagonal
    import DomainSets: ℝ

    @testset "BonnetStrategy with automatic conversion" begin
        a = MvNormalMeanCovariance([2.0, -3.0], Diagonal([1.5, 0.8]))
        my_logpdf(x) = logpdf(a, x)
        my_mv_continuous_logpdf = ContinuousMultivariateLogPdf(ℝ^2, my_logpdf)

        # Test with BonnetStrategy
        params = ProjectionParameters(
            niterations = 1000,
            strategy = BonnetStrategy(nsamples = 100),
        )
        prj = ProjectedTo(MvNormalMeanCovariance, 2; parameters = params)

        test_bonnet_proj = project_to(prj, my_mv_continuous_logpdf)
        @test test_bonnet_proj isa MvNormalMeanCovariance
        # BonnetStrategy is stochastic, so we use a larger tolerance
        @test mean(test_bonnet_proj) ≈ mean(a) atol = 0.5
        @test cov(test_bonnet_proj) ≈ cov(a) atol = 0.5
    end
end
