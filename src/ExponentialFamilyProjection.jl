module ExponentialFamilyProjection

using ExponentialFamily,
    ExponentialFamilyManifolds,
    BayesBase,
    Distributions,
    ManifoldsBase,
    Manifolds,
    Static,
    StatsFuns,
    LinearAlgebra,
    FastCholesky,
    Bumper,
    StaticArrays,
    Random

import BayesBase: InplaceLogpdf

# This can go to the ExponentialFamily.jl, very useful
# The idea here is that it is not necessary to pack a tuple of numbers into a vector
__projection_fast_pack_parameters(t::NTuple{N,<:Number}) where {N} = t
__projection_fast_pack_parameters(t) = ExponentialFamily.pack_parameters(t)

include("manopt/bounded_norm_update_rule.jl")
include("manopt/projection_objective.jl")
include("projected_to.jl")

### jacobian from natural to manifold
include("jacobians.jl")

"""
    preprocess_strategy_argument(strategy, argument)

Checks the compatibility of `strategy` with `argument` and returns a modified strategy and argument if needed.
"""
function preprocess_strategy_argument end

"""
    create_state!(
        strategy,
        M::AbstractManifold,
        parameters::ProjectionParameters,
        projection_argument,
        initial_ef,
        supplementary_η,
    )

Creates, initializes and returns a state for the `strategy` with the given parameters.
"""
function create_state! end

"""
    prepare_state!(
        strategy,
        state,
        M::AbstractManifold,
        parameters::ProjectionParameters,
        projection_argument,
        distribution,
        supplementary_η,
    )

Prepares an existing `state` of the `strategy` for the new optimization iteration for use by setting or updating its internal parameters.
"""
function prepare_state! end

"""
    compute_cost(
        M::AbstractManifold,
        strategy,
        state,
        η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )

Compute the cost using the provided `strategy`.

# Arguments
- `M::AbstractManifold`: The manifold on which the computations are performed.
- `strategy`: The strategy used for computation of the cost value.
- `state`: The current state for the `strategy`.
- `η`: Parameter vector.
- `logpartition`: The log partition of the current point (η).
- `gradlogpartition`: The gradient of the log partition of the current point (η).
- `inv_fisher`: The inverse Fisher information matrix of the current point (η).

# Returns
- `cost`: The computed cost value.
"""
function compute_cost end

"""
    compute_gradient!(
        M::AbstractManifold,
        strategy,
        state,
        X,
        η,
        logpartition,
        gradlogpartition,
        inv_fisher,
    )

Updates the gradient `X` in-place using the provided `strategy`.

# Arguments
- `M::AbstractManifold`: The manifold on which the computations are performed.
- `strategy`: The strategy used for computation of the gradient value.
- `state`: The current state of the control variate strategy.
- `X`: The storage for the gradient.
- `η`: Parameter vector.
- `logpartition`: The log partition of the current point (η).
- `gradlogpartition`: The gradient of the log partition of the current point (η).
- `inv_fisher`: The inverse Fisher information matrix of the current point (η).

# Returns
- `X`: The computed gradient (updated in-place)
"""
function compute_gradient! end

include("strategies/control_variate.jl")
include("strategies/mle.jl")
include("strategies/default.jl")
# Bonnet strategy
include("strategies/bonnet/bonnet_logpdf.jl")

end
