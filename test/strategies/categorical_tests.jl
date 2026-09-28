@testitem "Categorical analytic inverse Fisher action" begin
    using LinearAlgebra, StableRNGs, Random
    import ExponentialFamilyProjection: categorical_inv_fisher_mul!

    rng = StableRNG(42)
    for T in (Float32, Float64),
        probabilities in ([0.3, 0.7], [0.1, 0.2, 0.3, 0.4], [0.85, 0.1, 0.049, 0.001])

        p = T.(probabilities)
        p ./= sum(p)
        reduced_p = p[1:(end-1)]
        fisher = Diagonal(reduced_p) - reduced_p * reduced_p'
        v = randn(rng, T, length(reduced_p))
        expected = fisher \ v
        out = similar(v)
        tolerance = T === Float32 ? 2e-4 : 1e-10

        @test categorical_inv_fisher_mul!(out, p, v) === out
        @test out ≈ expected rtol = tolerance
        aliased = copy(v)
        @test categorical_inv_fisher_mul!(aliased, p, aliased) === aliased
        @test aliased ≈ expected rtol = tolerance
    end
end

@testitem "Categorical control variate gradient matches reduced Fisher solves" begin
    using BayesBase, Bumper, ExponentialFamily, ExponentialFamilyManifolds, LinearAlgebra
    import ExponentialFamilyProjection:
        ControlVariateStrategy, create_state!, compute_gradient!

    for p in ([0.3, 0.7], [0.1, 0.2, 0.3, 0.4], [0.85, 0.1, 0.049, 0.001]),
        buffer in (nothing, Bumper.SlabBuffer())

        K = length(p)
        n = K - 1
        ef = convert(ExponentialFamilyDistribution, Categorical(p))
        η = getnaturalparameters(ef)
        M = ExponentialFamilyManifolds.get_natural_manifold(Categorical, (), K)
        strategy = ControlVariateStrategy(nsamples = 37, buffer = buffer)
        parameters = ProjectionParameters()
        target = x -> -0.3 * x^2
        state = create_state!(strategy, M, parameters, target, ef, ())
        statistics_before = copy(state.sufficientstatistics)
        gradsamples_before = copy(state.gradsamples)
        μ = gradlogpartition(ef)
        reduced_p = μ[1:n]
        fisher = Diagonal(reduced_p) - reduced_p * reduced_p'
        statistics = state.sufficientstatistics[1:n, :]
        gradients = state.gradsamples[1:n, :]
        C = cov(statistics', gradients')
        δ = vec(mean(statistics, dims = 2)) - reduced_p
        g = vec(mean(gradients, dims = 2))

        # Also exercise the shifted η used for supplementary distributions.
        for supplementary_η in (zeros(K), vcat(fill(0.2, n), 0.0))
            shifted_η = η - supplementary_η
            expected = shifted_η[1:n] - fisher \ (g - C * (fisher \ δ))
            X = fill(NaN, K)
            @test compute_gradient!(
                M,
                strategy,
                state,
                X,
                shifted_η,
                logpartition(ef),
                μ,
                nothing,
            ) === X
            @test X[1:n] ≈ expected rtol = 1e-10
            @test iszero(X[end])

            aliased = copy(shifted_η)
            compute_gradient!(
                M,
                strategy,
                state,
                aliased,
                aliased,
                logpartition(ef),
                μ,
                nothing,
            )
            @test aliased ≈ X
        end
        @test state.sufficientstatistics == statistics_before
        @test state.gradsamples == gradsamples_before
    end
end

@testitem "Categorical gradient agrees with exact expectation gradient" begin
    using ExponentialFamily, ExponentialFamilyManifolds, LinearAlgebra
    import ExponentialFamilyProjection:
        ControlVariateStrategy, ControlVariateStrategyState, compute_gradient!

    K = 4
    p = fill(1 / K, K)
    η = zeros(K)
    ℓ = log.([0.1, 0.2, 0.3, 0.4])
    statistics = Matrix{Float64}(I, K, K)
    # One sample per category integrates expectations exactly under uniform q.
    state = ControlVariateStrategyState(
        samples = collect(1:K),
        logpdfs = ℓ,
        logbasemeasures = zeros(K),
        sufficientstatistics = statistics,
        gradsamples = (statistics .- p) .* ℓ',
    )
    M = ExponentialFamilyManifolds.get_natural_manifold(Categorical, (), K)
    X = similar(η)
    compute_gradient!(M, ControlVariateStrategy(), state, X, η, log(K), p, nothing)
    @test X ≈ η - (ℓ .- ℓ[end])
end

@testitem "Categorical objective uses only sampled target evaluations" begin
    using BayesBase, ExponentialFamily, ExponentialFamilyManifolds, Manifolds
    import ExponentialFamilyProjection:
        ControlVariateStrategy,
        ProjectionCostGradientObjective,
        create_state!,
        compute_gradient!,
        prepare_inv_fisher

    K = 100
    p = vcat(0.9, fill(0.1 / (K - 1), K - 1))
    ef = convert(ExponentialFamilyDistribution, Categorical(p))
    η = getnaturalparameters(ef)
    M = ExponentialFamilyManifolds.get_natural_manifold(Categorical, (), K)
    point = ExponentialFamilyManifolds.partition_point(Categorical, (), copy(η), K)
    strategy = ControlVariateStrategy(nsamples = 12)
    parameters = ProjectionParameters(strategy = strategy)
    evaluated = Int[]
    target = x -> begin
        push!(evaluated, x)
        -log1p(x)
    end
    state = create_state!(strategy, M, parameters, target, ef, ())
    objective = ProjectionCostGradientObjective(
        parameters,
        target,
        copy(point),
        (),
        strategy,
        state,
    )
    @test prepare_inv_fisher(M, strategy, ef) === nothing
    empty!(evaluated)
    X = zero_vector(M, point)
    cost, gradient = objective(M, X, point)
    @test evaluated == state.samples
    @test length(evaluated) == strategy.nsamples
    @test isfinite(cost)
    @test gradient === X
    @test all(isfinite, X)
    @test iszero(X[end])

    expected = zeros(K)
    compute_gradient!(
        M,
        strategy,
        state,
        expected,
        η,
        logpartition(ef),
        gradlogpartition(ef),
        nothing,
    )
    @test collect(X) ≈ expected
end

@testitem "Categorical MLE matches autodiff and reduced Fisher solves" begin
    using BayesBase,
        ExponentialFamily, ExponentialFamilyManifolds, ForwardDiff, LinearAlgebra, Manifolds
    import ExponentialFamilyProjection:
        MLEStrategy,
        ProjectionCostGradientObjective,
        create_state!,
        compute_gradient!,
        prepare_inv_fisher,
        gettargetfn

    for p in ([0.3, 0.7], [0.1, 0.2, 0.3, 0.4], [0.85, 0.1, 0.049, 0.001])
        K = length(p)
        n = K - 1
        ef = convert(ExponentialFamilyDistribution, Categorical(p))
        η = getnaturalparameters(ef)
        μ = gradlogpartition(ef)
        M = ExponentialFamilyManifolds.get_natural_manifold(Categorical, (), K)
        point = ExponentialFamilyManifolds.partition_point(Categorical, (), copy(η), K)
        strategy = MLEStrategy()
        parameters = ProjectionParameters(strategy = strategy)
        # Include absent categories, including an absent reference category.
        for samples in (collect(1:K), [1, 1, 2, K], [1, 1, 1])
            state = create_state!(strategy, M, parameters, samples, ef, ())
            statistics_before = copy(gettargetfn(state).sufficientstatistics)
            @test prepare_inv_fisher(M, strategy, ef) === nothing
            reduced_p = μ[1:n]
            fisher = Diagonal(reduced_p) - reduced_p * reduced_p'

            for supplementary in ((), (vcat(fill(0.2, n), 0.0),))
                shifted_η = isempty(supplementary) ? η : η - only(supplementary)
                ordinary_gradient = ForwardDiff.gradient(shifted_η[1:n]) do reduced_η
                    gettargetfn(state)(vcat(reduced_η, zero(eltype(reduced_η))))
                end
                expected = fisher \ ordinary_gradient
                X = fill(NaN, K)
                @test compute_gradient!(
                    M,
                    strategy,
                    state,
                    X,
                    shifted_η,
                    logpartition(ef),
                    μ,
                    nothing,
                ) === X
                @test X[1:n] ≈ expected rtol = 1e-10
                @test iszero(X[end])

                aliased = copy(shifted_η)
                compute_gradient!(
                    M,
                    strategy,
                    state,
                    aliased,
                    aliased,
                    logpartition(ef),
                    μ,
                    nothing,
                )
                @test aliased ≈ X

                objective = ProjectionCostGradientObjective(
                    parameters,
                    samples,
                    copy(point),
                    supplementary,
                    strategy,
                    state,
                )
                tangent = zero_vector(M, point)
                cost, result = objective(M, tangent, point)
                @test result === tangent
                @test collect(tangent) ≈ X
                @test cost ≈ gettargetfn(state)(shifted_η)
            end
            @test gettargetfn(state).sufficientstatistics == statistics_before
        end
    end
end
