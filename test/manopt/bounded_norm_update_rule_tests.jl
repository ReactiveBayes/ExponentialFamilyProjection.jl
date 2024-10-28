@testitem "`BoundedNormUpdateRule` should bound the gradient to a specific value" begin
    using Manopt, Manifolds, StableRNGs, Static, JET

    import ExponentialFamilyProjection: BoundedNormUpdateRule

    rng = StableRNG(42)
    M = Euclidean(3)

    f(M, p) = norm(p)^2
    grad_f(M, p) = 2 * p

    # The `apply_update_rules_for_test` function applies different update rules to the optimization problem and gradient descent state, 
    # returning the unbounded gradient as the first argument and a collection of bounded gradients for testing.
    function apply_update_rules_for_test(p, limit)
        cpa = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        gst = GradientDescentState(M; p=zero(p))
        Manopt.set_iterate!(gst, M, p)

        _, X = Manopt.IdentityUpdateRule()(cpa, gst, 1)
        X_identity = copy(X)

        _, X = BoundedNormUpdateRule(limit)(cpa, gst, 1)
        X_bounded_with_float = copy(X)

        _, X = BoundedNormUpdateRule(static(limit))(cpa, gst, 1)
        X_bounded_with_static = copy(X)

        _, X = BoundedNormUpdateRule(Inf; direction = BoundedNormUpdateRule(limit))(
            cpa,
            gst,
            1,
        )
        X_bounded_twice = copy(X)

        X_bounded = (X_bounded_with_float, X_bounded_with_static, X_bounded_twice)

        return X_identity, X_bounded
    end

    @testset "`X` is above the limit" begin
        # The points in `pts` should produce the gradient, which is above the limit
        pts = (
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 1.0],
            [100.0, 1.0, 100.0],
            [1.0, 100.0, 100.0],
        )
        for limit in (10, 5, 1, 0.1), p in pts
            X_identity, X_bounded = apply_update_rules_for_test(p, limit)

            @test norm(M, p, X_identity) > limit

            for X in X_bounded
                @test norm(M, p, X) ≈ limit
            end
        end
    end

    @testset "`X` is below the limit" begin
        # The points in `pts` should produce the gradient, which is below the limit
        pts = (
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.0],
            [0.1, 0.0, 0.1],
            [0.0, 0.1, 0.1],
        )
        for limit in (10, 5, 1), p in pts
            X_identity, X_bounded = apply_update_rules_for_test(p, limit)

            @test norm(M, p, X_identity) < limit

            for X in X_bounded
                @test norm(M, p, X) <= limit && norm(M, p, X) == norm(M, p, X_identity)
            end
        end
    end

    @testset "JET tests" begin
        for limit in (1, 1.0, 1.0f0), p in (zeros(Float64, 3), zeros(Float32, 3))
            cpa = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            gst = GradientDescentState(M; p=zero(p))
            @test_opt BoundedNormUpdateRule(limit)(cpa, gst, 1)
            @test_opt BoundedNormUpdateRule(static(limit))(cpa, gst, 1)
            @test_opt BoundedNormUpdateRule(
                static(limit);
                direction = BoundedNormUpdateRule(limit),
            )(
                cpa,
                gst,
                1,
            )
        end
    end

end