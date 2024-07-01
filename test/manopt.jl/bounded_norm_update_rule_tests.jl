@testitem "`BoundedNormUpdateRule` should bound the gradient to a specific value" begin
    using Manopt, Manifolds, StableRNGs, Static, JET

    import ExponentialFamilyProjection: BoundedNormUpdateRule

    rng = StableRNG(42)
    M = Euclidean(3)
    f(M, p) = norm(p)^2
    grad_f(M, p) = 2 * p
    cpa = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))

    @testset "`X` is above the limit" begin
        p = [100.0, 100.0, 100.0]
        gst = GradientDescentState(M, zero(p))

        Manopt.set_iterate!(gst, M, p)

        for limit in (10, 5, 1, 0.1)

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
            X_bounded_two_times = copy(X)

            @test norm(M, p, X_identity) > limit
            @test norm(M, p, X_bounded_with_float) ≈ limit
            @test norm(M, p, X_bounded_with_static) ≈ limit
            @test norm(M, p, X_bounded_two_times) ≈ limit

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

    @testset "`X` is below the limit" begin
        p = [0.1, 0.1, 0.1]
        gst = GradientDescentState(M, zero(p))

        Manopt.set_iterate!(gst, M, p)

        for limit in (10, 5, 1)

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
            X_bounded_two_times = copy(X)

            @test norm(M, p, X_identity) < limit
            @test norm(M, p, X_bounded_with_float) <= limit && norm(M, p, X_bounded_with_float) == norm(M, p, X_identity)
            @test norm(M, p, X_bounded_with_static) <= limit && norm(M, p, X_bounded_with_static) == norm(M, p, X_identity)
            @test norm(M, p, X_bounded_two_times) <= limit && norm(M, p, X_bounded_two_times) == norm(M, p, X_identity)
        end
    end

end