using ExponentialFamilyProjection, Test, ReTestItems, Random

# Set the random seed for reproducibility of kl-divergence tests
Random.seed!(42)

runtests(ExponentialFamilyProjection; memory_threshold = 1.0)
