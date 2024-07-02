@testitem "`BoundedNormUpdateRule` should bound the gradient to a specific value" begin
    using Manopt, Manifolds, StableRNGs, Static, JET

    import ExponentialFamilyProjection: BoundedNormUpdateRule

    rng = StableRNG(42)
    M = Euclidean(3)

    f(M, p) = norm(p)^2
    grad_f(M, p) = 2 * p

    # The `apply_update_rules_for_test` function applies different update rules to the optimization problem and gradient descent state, 
    # returning the resulting points for testing.
    function apply_update_rules_for_test(p, limit)
        cpa = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        gst = GradientDescentState(M, zero(p))
        Manopt.set_iterate!(gst, M, p)

        _, X = IdentityUpdateRule()(cpa, gst, 1)
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
        return X_identity, X_bounded_with_float, X_bounded_with_static, X_bounded_twice
    end

    @testset "`X` is above the limit" begin
        # The point `p` should produce the gradient, which is above the limit
        p = [100.0, 100.0, 100.0]

        for limit in (10, 5, 1, 0.1)

            X_identity, X_bounded_with_float, X_bounded_with_static, X_bounded_twice =
                apply_update_rules_for_test(p, limit)

            @test norm(M, p, X_identity) > limit
            @test norm(M, p, X_bounded_with_float) ≈ limit
            @test norm(M, p, X_bounded_with_static) ≈ limit
            @test norm(M, p, X_bounded_twice) ≈ limit

        end
    end

    @testset "`X` is below the limit" begin
        # The point `p` should produce the gradient, which is below the limit
        p = [0.1, 0.1, 0.1]

        for limit in (10, 5, 1)

            X_identity, X_bounded_with_float, X_bounded_with_static, X_bounded_twice =
                apply_update_rules_for_test(p, limit)

            @test norm(M, p, X_identity) < limit
            @test norm(M, p, X_bounded_with_float) <= limit &&
                  norm(M, p, X_bounded_with_float) == norm(M, p, X_identity)
            @test norm(M, p, X_bounded_with_static) <= limit &&
                  norm(M, p, X_bounded_with_static) == norm(M, p, X_identity)
            @test norm(M, p, X_bounded_twice) <= limit &&
                  norm(M, p, X_bounded_twice) == norm(M, p, X_identity)
        end
    end

    @testset "JET tests" begin
        for limit in (1, 1.0, 1.0f0), p in (zeros(Float64, 3), zeros(Float32, 3))
            cpa = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            gst = GradientDescentState(M, zero(p))
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