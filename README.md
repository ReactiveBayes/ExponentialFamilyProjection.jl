# ExponentialFamilyProjection

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://reactivebayes.github.io/ExponentialFamilyProjection.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/ExponentialFamilyProjection.jl/dev/)
[![Build Status](https://github.com/reactivebayes/ExponentialFamilyProjection.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/reactivebayes/ExponentialFamilyProjection.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/reactivebayes/ExponentialFamilyProjection.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/reactivebayes/ExponentialFamilyProjection.jl)

The `ExponentialFamilyProjection.jl` package offers a suite of functions for projecting an arbitrary (un-normalized) log probability density function onto a specified member of the exponential family (e.g., Gaussian, Beta, Bernoulli). This is achieved by optimizing the natural parameters of the exponential family member within a defined manifold. The library leverages `Manopt.jl` for optimization and utilizes [`ExponentialFamilyManifolds.jl`](https://github.com/ReactiveBayes/ExponentialFamilyManifolds.jl) to define the manifolds corresponding to the members of the exponential family.

For detailed information and usage examples, please refer to the [documentation](https://reactivebayes.github.io/ExponentialFamilyProjection.jl/stable/).