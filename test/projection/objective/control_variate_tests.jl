@testitem "The CVI objective cost value with supplementary natural parameters should approximatly equal to just `ProductOf`" begin
    using ExponentialFamily, Distributions, BayesBase
    using Random, StableRNGs
    using Manopt, ExponentialFamilyManifolds, ExponentialFamilyProjection

    # This distribution are selected because their log base measure is zero
    # Once it's not zero the test will fail
    distributions = [
        (Bernoulli(0.7), Bernoulli(0.1)),
        (Bernoulli(0.1), Bernoulli(0.9)),
        (Bernoulli(0.9), Bernoulli(0.1)),
        (Beta(10, 10), Beta(3, 3)),
        (Beta(1, 1), Beta(0.1, 3)),
        (Gamma(1, 1), Gamma(10, 10)),
    ]

    for distribution in distributions
        @testset let left = distribution[1], right = distribution[2], nsamples = 2000, npoints = 20
            dims = size(rand(left))
            
            typetag = ExponentialFamily.exponential_family_typetag(left)
            
            manifold = ExponentialFamilyManifolds.get_natural_manifold(
                typetag,
                dims,
                nothing
            )
    
            targetfn_part = (x) -> logpdf(left, x)
            targetfn_full = (x) -> logpdf(ProductOf(left, right), x)
            ef = convert(ExponentialFamilyDistribution, right)
            supplementary_ef = [getnaturalparameters(convert(ExponentialFamilyDistribution, right)),]

            seeds = rand(StableRNG(42), UInt, npoints)

            point = rand(StableRNG(42), manifold)

            costs = map(seeds) do seed
                obj_part = ExponentialFamilyProjection.CVICostGradientObjective(
                    targetfn_part,
                    supplementary_ef,
                    ExponentialFamilyProjection.ControlVariateStrategy(nsamples = nsamples, seed = seed),
                    nothing
                )

                obj_full = ExponentialFamilyProjection.CVICostGradientObjective(
                    targetfn_full,
                    [],
                    ExponentialFamilyProjection.ControlVariateStrategy(nsamples = nsamples, seed = seed),
                    nothing
                )

                rng = StableRNG(42)
                
                X2 = Manopt.zero_vector(manifold, point)
                X1 = Manopt.zero_vector(manifold, point)

                c1, _ = obj_part(manifold, X1, point)
                c2, _ = obj_full(manifold, X2, point)
            
                [c1, c2]
            end

            c1s = map(costs) do c
                c[1]
            end

            c2s = map(costs) do c
                c[2]
            end

            mean_c1, std_c1 = mean(c1s) .+ logpartition(ef), std(c1s)
            mean_c2, std_c2 = mean(c2s), std(c2s)
            
            upper_bound_c1 = mean_c1 + 3 * std_c1
            lower_bound_c1 = mean_c1 - 3 * std_c1

            upper_bound_c2 = mean_c2 + 3 * std_c2
            lower_bound_c2 = mean_c2 - 3 * std_c2

            @test lower_bound_c1 < upper_bound_c2 && upper_bound_c1 > lower_bound_c2
        end
    end
end