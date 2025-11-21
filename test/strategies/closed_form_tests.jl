module ClosedFormStrategyTests

using Test
using ExponentialFamilyProjection
using ExponentialFamilyProjection: ControlVariateStrategy
using ClosedFormExpectations
using Distributions
using ExponentialFamily
using StableRNGs
using LinearAlgebra

@testset "ClosedFormStrategy" begin

    @testset "LogGamma projected to Normal" begin
        # Target: LogGamma(α=2.0, β=1.0)
        # Density: p(x) ∝ x^(α-1) * e^(-βx)
        # Log density: (α-1)log(x) - βx
        
        # Note: Normal support is (-∞, ∞), Gamma is (0, ∞). 
        # This projection is technically improper if Normal puts mass on x < 0.
        # But we can test if the gradient machinery works.
        # To make it "reasonable", we put the Normal far from 0.
        
        target_dist = LogGamma(20.0, 1.0) # Mean 20, Var 20.
        target = Logpdf(target_dist)
        
        # Initial approximation: Normal(15, 5)
        initial_dist = Normal(15.0, 5.0)
        
        # Create strategy
        strategy = ClosedFormStrategy()
        
        # Project
        # We use a small number of iterations to check it runs and descends
        result = project_to(ProjectedTo(NormalMeanVariance), target; 
            strategy = strategy, 
            initial_point = initial_dist,
            parameters = ProjectionParameters(niterations = 50, tolerance = 1e-5)
        )
        
        @test result isa NormalMeanVariance
        μ, v = mean(result), var(result)

        # We expect it to move towards the mode of the LogGamma.
        # Mode of LogGamma(α, β) is at x = log(α*β).
        # Here α=20, β=1. Mode at log(20) ≈ 3.0.

        @test 2.0 < μ < 4.0 
    end

    @testset "LogNormal projected to Gamma" begin
        # Target: LogNormal(μ=1.0, σ=0.5)
        # Approx: Gamma
        
        target_dist = LogNormal(1.0, 0.5)
        target = Logpdf(target_dist)
        
        # Initial: Gamma(2.0, 2.0) -> mean 4, var 8
        initial_dist = Gamma(2.0, 2.0)
        
        strategy = ClosedFormStrategy()
        
        result = project_to(ProjectedTo(Gamma), target;
            strategy = strategy,
            initial_point = initial_dist,
            parameters = ProjectionParameters(niterations = 50, tolerance = 1e-5)
        )
        
        @test result isa GammaDistributionsFamily
        
        # Check if it ran without error and produced valid parameters
        @test shape(result) > 0
        @test scale(result) > 0
        
        # Comparison with ControlVariateStrategy
        # CV strategy is stochastic, so it might be slightly different but should be close.
        
        cv_strategy = ControlVariateStrategy(nsamples = 500)
        result_cv = project_to(ProjectedTo(Gamma), target;
            strategy = cv_strategy,
            initial_point = initial_dist,
            parameters = ProjectionParameters(niterations = 50, tolerance = 1e-4)
        )
        
        # KL between result and result_cv should be small
        # Since they are both Gamma, we can measure parameter distance or KL
        
        @test isapprox(mean(result), mean(result_cv), rtol=0.1)
    end

    @testset "Comparison: Speed and Accuracy" begin
        # Simple case: Normal to Normal (should be exact one step if Newton, but we use GradientDescent)
        # Let's use a target that is actually a Normal
        target_dist = Normal(5.0, 2.0)
        target = Logpdf(target_dist)
        
        initial = Normal(0.0, 1.0)
        
        # Analytic
        t_analytic = @elapsed begin
            res_analytic = project_to(ProjectedTo(NormalMeanVariance), target;
                strategy = ClosedFormStrategy(),
                initial_point = initial,
                parameters = ProjectionParameters(niterations=100)
            )
        end
        
        # MC
        t_mc = @elapsed begin
            res_mc = project_to(ProjectedTo(NormalMeanVariance), target;
                strategy = ControlVariateStrategy(nsamples=1000),
                initial_point = initial,
                parameters = ProjectionParameters(niterations=100)
            )
        end
        
        println("Analytic time: $t_analytic")
        println("MC time: $t_mc")
        
        # Analytic should be faster (no sampling) and more accurate (converge to exact target)
        @test isapprox(mean(res_analytic), mean(target_dist), atol=2e-2)
        @test isapprox(std(res_analytic), std(target_dist), atol=2e-2)
        
        # MC might have some noise
        @test abs(mean(res_analytic) - 5.0) < abs(mean(res_mc) - 5.0) + 0.1 # Heuristic check
    end

end

end




