@testitem "`MLEStrategy` should fail if given a function instead of a list of sampls" begin
    using ExponentialFamily

    prj = ProjectedTo(
        Beta;
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.MLEStrategy(),
        ),
    )

    @test_throws "`MLEStrategy` requires the projection argument to be an array of samples." project_to(
        prj,
        (x) -> 1,
    )

end

@testitem "`MLEStrategy` target function should compute the mean of logpdfs given samples" begin
    using ExponentialFamily,
        LinearAlgebra,
        Distributions,
        ExponentialFamilyManifolds,
        ExponentialFamilyProjection,
        StableRNGs,
        ForwardDiff,
        Manifolds,
        FastCholesky

    rng = StableRNG(42)
    for distribution in [
            Bernoulli(0.5),
            Beta(1, 1),
            NormalMeanVariance(0, 1),
            # ForwardDiff gradient is not correct for MvNormalMeanCovariance
            # So we will not compare out closed form gradient with ForwardDiff gradient
            # MvNormalMeanCovariance(ones(2), Matrix(Diagonal(ones(2)))),
            Poisson(0.5),
            Chisq(30.0),
            Gamma(1, 1),
        ],
        nsamples in (100, 500)

        ef = convert(ExponentialFamilyDistribution, distribution)
        samples = rand(rng, ef, nsamples)

        T = ExponentialFamily.exponential_family_typetag(ef)
        c = getconditioner(ef)
        d = size(rand(rng, ef))
        M = ExponentialFamilyManifolds.get_natural_manifold(T, d, c)
        proj_params = ProjectionParameters()
        η = getnaturalparameters(ef)

        strategy = ExponentialFamilyProjection.MLEStrategy()
        state = ExponentialFamilyProjection.create_state!(
            strategy,
            M,
            proj_params,
            samples,
            ef,
            (),
        )
        obj = ExponentialFamilyProjection.ProjectionCostGradientObjective(
            proj_params,
            samples,
            copy(η),
            (),
            strategy,
            state,
        )

        @test state == state

        _logpartition = logpartition(ef)
        _gradlogpartition = gradlogpartition(ef)
        _inv_fisher = cholinv(fisherinformation(ef))
        cost = ExponentialFamilyProjection.compute_cost(
            M,
            strategy,
            state,
            η,
            _logpartition,
            _gradlogpartition,
            _inv_fisher,
        )

        gradient = similar(η)

        ExponentialFamilyProjection.compute_gradient!(
            M,
            strategy,
            state,
            gradient,
            η,
            _logpartition,
            _gradlogpartition,
            _inv_fisher,
        )

        _, samples_container = ExponentialFamily.check_logpdf(ef, samples)
        expected_cost = -mean(logpdf(ef, samples))
        expected_gradient = ForwardDiff.gradient(η) do η_new
            p_new = ExponentialFamilyManifolds.partition_point(M, η_new)
            ef_new = convert(ExponentialFamilyDistribution, M, p_new)
            return -mean(logpdf(ef_new, samples))
        end

        @test cost ≈ expected_cost
        @test gradient ≈ (_inv_fisher * expected_gradient)

        # Test gradient computation in manifold coordinates
        p = ExponentialFamilyManifolds.partition_point(M, η)
        expected_gradient_p = ForwardDiff.gradient(p) do p_new
            ef_new = convert(ExponentialFamilyDistribution, M, p_new)
            return -mean(logpdf(ef_new, samples))
        end
        X_p = Manifolds.zero_vector(M, p)
        c_p, X_p = obj(M, X_p, p)
        @test c_p ≈ cost
        if distribution isa Gamma
            @test_broken X_p ≈ _inv_fisher * expected_gradient_p
        else
            @test X_p ≈ _inv_fisher * expected_gradient_p
        end
    end
end
