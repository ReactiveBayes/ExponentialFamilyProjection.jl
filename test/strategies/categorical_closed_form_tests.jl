@testitem "Categorical closed-form cost and natural gradient" begin
    using ClosedFormExpectations, ExponentialFamily, ExponentialFamilyManifolds
    using LinearAlgebra
    import ExponentialFamilyProjection as EFP

    for T in (Float32, Float64),
        probabilities in ([0.3, 0.7], [0.1, 0.2, 0.3, 0.4], [0.85, 0.1, 0.049, 0.001])

        p = T.(probabilities)
        p ./= sum(p)
        K = length(p)
        ef = convert(ExponentialFamilyDistribution, Categorical(p))
        η = copy(getnaturalparameters(ef))
        M = ExponentialFamilyManifolds.get_natural_manifold(Categorical, (), K)
        strategy = ClosedFormStrategy()
        parameters = ProjectionParameters(; strategy)
        scores = T.([0.3k - 0.2k^2 for k = 1:K])
        target = k -> scores[k]
        initial_ef = convert(ExponentialFamilyDistribution, Categorical(fill(T(1 / K), K)))
        state = EFP.create_state!(strategy, M, parameters, target, initial_ef, ())
        @test EFP.prepare_state!(strategy, state, M, parameters, target, ef, ()) === state
        @test EFP.prepare_inv_fisher(M, strategy, ef) === nothing

        μ = ExponentialFamily.gradlogpartition(ef)
        A = ExponentialFamily.logpartition(ef)
        tolerance = T === Float32 ? 2e-3 : 1e-10
        for supplementary_η in (zeros(T, K), T.(vcat(fill(0.2, K - 1), 0.0)), copy(η))
            shifted_η = η - supplementary_η
            X = fill(T(NaN), K)
            @test EFP.compute_gradient!(M, strategy, state, X, shifted_η, A, μ, nothing) ===
                  X

            # Independent dense solve in the nonsingular reference chart.
            F = Diagonal(μ[1:(K-1)]) - μ[1:(K-1)] * μ[1:(K-1)]'
            score_gradient = μ .* (scores .- dot(μ, scores))
            expected = shifted_η[1:(K-1)] - F \ score_gradient[1:(K-1)]
            @test X[1:(K-1)] ≈ expected rtol = tolerance atol = tolerance
            # Exact categorical KL has a linear natural gradient in logits.
            @test X[1:(K-1)] ≈ shifted_η[1:(K-1)] - (scores[1:(K-1)] .- scores[K]) rtol =
                tolerance atol = tolerance
            @test iszero(X[end])

            cost = EFP.compute_cost(M, strategy, state, shifted_η, A, μ, nothing)
            expected_cost = sum(μ .* (log.(μ) .- scores .- supplementary_η))
            @test cost ≈ expected_cost rtol = tolerance atol = tolerance

            aliased = copy(shifted_η)
            EFP.compute_gradient!(M, strategy, state, aliased, aliased, A, μ, nothing)
            @test aliased ≈ X rtol = tolerance atol = tolerance
        end
    end
end

@testitem "Categorical closed-form objective finite differences with supplementary factors" begin
    using ClosedFormExpectations, ExponentialFamily, ExponentialFamilyManifolds
    using LinearAlgebra
    using Manifolds: zero_vector
    import ExponentialFamilyProjection as EFP

    p = [0.15, 0.25, 0.6]
    ef = convert(ExponentialFamilyDistribution, Categorical(p))
    η = copy(getnaturalparameters(ef))
    M = ExponentialFamilyManifolds.get_natural_manifold(Categorical, (), 3)
    strategy = ClosedFormStrategy()
    parameters = ProjectionParameters(; strategy)
    target = k -> sin(k) - 0.2k
    supplementary = (
        getnaturalparameters(
            convert(ExponentialFamilyDistribution, Categorical([0.6, 0.1, 0.3])),
        ),
        getnaturalparameters(
            convert(ExponentialFamilyDistribution, Categorical([0.2, 0.5, 0.3])),
        ),
    )
    state = EFP.create_state!(strategy, M, parameters, target, ef, supplementary)
    objective = EFP.ProjectionCostGradientObjective(
        parameters,
        target,
        copy(η),
        supplementary,
        strategy,
        state,
    )
    point = ExponentialFamilyManifolds.partition_point(M, copy(η))
    X = zero_vector(M, point)
    cost, _ = EFP.call_objective(objective, M, X, point)
    reduced_F = Diagonal(p[1:2]) - p[1:2] * p[1:2]'
    ordinary_gradient = reduced_F * X[1:2]
    for i = 1:2
        plus, minus = copy(η), copy(η)
        plus[i] += 1e-5
        minus[i] -= 1e-5
        plus_point = ExponentialFamilyManifolds.partition_point(M, plus)
        minus_point = ExponentialFamilyManifolds.partition_point(M, minus)
        cp, _ = EFP.call_objective(objective, M, zero_vector(M, plus_point), plus_point)
        cm, _ = EFP.call_objective(objective, M, zero_vector(M, minus_point), minus_point)
        @test ordinary_gradient[i] ≈ (cp - cm) / 2e-5 atol = 1e-9
    end
    @test getnaturalparameters(ef) == η
    @test iszero(X[end])
end

@testitem "Categorical closed-form public projection" begin
    using ClosedFormExpectations, ExponentialFamily, ExponentialFamilyManifolds, Manopt
    using BayesBase: ProductOf
    using Distributions: probs
    using StatsFuns: softmax
    import ExponentialFamilyProjection as EFP

    struct CategoryLogScore{T}
        scores::T
    end
    (f::CategoryLogScore)(k) = f.scores[k]

    function projection(K; seed = 42)
        ProjectedTo(
            Categorical;
            conditioner = K,
            parameters = ProjectionParameters(
                strategy = ClosedFormStrategy(),
                niterations = 1,
                tolerance = missing,
                stepsize = Manopt.ConstantLength(1.0),
                direction = Manopt.IdentityUpdateRule(),
                seed = seed,
            ),
        )
    end

    for p in ([0.8, 0.2], [0.2, 0.3, 0.5], [0.85, 0.1, 0.049, 0.001])
        K = length(p)
        oldq = Categorical(p)
        scores = [0.2k - 0.1k^2 for k = 1:K]
        expected = softmax(scores)
        target = Categorical(expected)
        for argument in
            (target, Logpdf(target), k -> scores[k], CategoryLogScore(scores), identity)
            wanted = argument === identity ? softmax(collect(1.0:K)) : expected
            result = project_to(projection(K), argument; initialpoint = oldq)
            @test probs(result) ≈ wanted atol = 1e-10
            @test iszero(
                getnaturalparameters(convert(ExponentialFamilyDistribution, result))[end],
            )
        end

        # Multiplication by supplementary densities, including the current policy.
        target_fn = k -> scores[k]
        for seed in (7, 42, 123)
            result = project_to(projection(K; seed), target_fn, oldq; initialpoint = oldq)
            @test probs(result) ≈ softmax(log.(p) + scores) atol = 1e-10
        end
        shifted = project_to(projection(K), k -> scores[k] + 5.0, oldq; initialpoint = oldq)
        @test probs(shifted) ≈ softmax(log.(p) + scores) atol = 1e-10
        constant = project_to(projection(K), _ -> 3.0, oldq; initialpoint = oldq)
        @test probs(constant) ≈ p atol = 1e-10

        second = Categorical(softmax(reverse(scores)))
        product_result =
            project_to(projection(K), ProductOf(target, second); initialpoint = oldq)
        @test probs(product_result) ≈ softmax(log.(probs(target)) + log.(probs(second))) atol =
            1e-10
        supplementary_result =
            project_to(projection(K), target_fn, oldq, second; initialpoint = oldq)
        @test probs(supplementary_result) ≈ softmax(log.(p) + scores + log.(probs(second))) atol =
            1e-10
    end

    @test_throws ArgumentError project_to(
        projection(2),
        [1, 2, 1];
        initialpoint = Categorical([0.5, 0.5]),
    )
    # Family-aware preprocessing must preserve a score closure, including captured arrays.
    M = ExponentialFamilyManifolds.get_natural_manifold(Categorical, (), 2)
    fn = let scores = [1.0, 2.0]
        k -> scores[k]
    end
    strategy = ClosedFormStrategy()
    @test EFP.preprocess_strategy_argument(M, strategy, fn) == (strategy, fn)
end

@testitem "Kakade exact directions through categorical closed-form projection" begin
    using ClosedFormExpectations, ExponentialFamily, Manopt
    using StatsFuns: logistic, logit

    rate = 0.05
    prj = ProjectedTo(
        Categorical;
        conditioner = 2,
        parameters = ProjectionParameters(
            strategy = ClosedFormStrategy(),
            niterations = 1,
            tolerance = missing,
            stepsize = Manopt.ConstantLength(1.0),
            direction = Manopt.IdentityUpdateRule(),
        ),
    )
    for θ in (logit.([0.8, 0.1]), [0.0, 0.0], [-2.0, 3.0], [12.0, 0.0], [2.0, 12.0])
        p, c = logistic.(θ), logistic.(-θ)
        d = reverse(c) ./ sum(c)
        w = d .* p .* c
        differences = [(2 - 3p[2]) / sum(c), (4 - 3p[1]) / sum(c)]
        for method in (:ordinary, :natural), s = 1:2
            factor = method == :natural ? w[s] / (w[s] + 1e-3) : w[s]
            scores = rate * factor .* [c[s] * differences[s], -p[s] * differences[s]]
            oldq = Categorical([p[s], c[s]])
            result = project_to(prj, k -> scores[k], oldq; initialpoint = oldq)
            new_θ = getnaturalparameters(convert(ExponentialFamilyDistribution, result))[1]
            @test new_θ - θ[s] ≈ rate * factor * differences[s] atol = 1e-10 rtol = 1e-8
        end
    end
end
