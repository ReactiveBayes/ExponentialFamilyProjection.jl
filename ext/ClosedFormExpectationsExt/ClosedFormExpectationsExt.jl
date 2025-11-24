module ClosedFormExpectationsExt

using ExponentialFamilyProjection
using ClosedFormExpectations
using ExponentialFamily
using ExponentialFamilyManifolds
using Manifolds
using ManifoldsBase
using LinearAlgebra
using Distributions

# Import types needed for RxInfer closure unwrapping
import ExponentialFamily: ProductOf
import ClosedFormExpectations: Logpdf

ExponentialFamilyProjection.get_nsamples(::ClosedFormStrategy) = 0

function logbasemeasure_correction(
    ::ClosedFormStrategy,
    ::ExponentialFamily.ConstantBaseMeasure,
    q_dist,
    grad_target,
)
    grad_target
end

function ExponentialFamilyProjection.compute_gradient!(
    M::AbstractManifold,
    strategy::ClosedFormStrategy,
    state,
    X,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    # The gradient of KL(q||p) involves E_q[(log p̃ - log h_q) * (T - μ)]
    # where h_q is the base measure of q (the variational distribution).
    #
    # For constant base measure:
    #   E[(log p̃ - log h) * (T - μ)] = E[log p̃ * (T - μ)] - log h * E[(T - μ)]
    #   Since E[T] = μ, the second term is zero.

    target_fn = state.target

    # Convert natural parameters on manifold to an ExponentialFamilyDistribution object
    q_dist = convert(
        ExponentialFamilyDistribution,
        M,
        ExponentialFamilyManifolds.partition_point(M, η),
    )

    # Compute ∇_η E[log p̃ * (T - μ)]
    grad_target = mean(ClosedWilliamsProduct(), target_fn, q_dist)
    grad_eta = logbasemeasure_correction(
        strategy,
        ExponentialFamily.isbasemeasureconstant(q_dist),
        q_dist,
        grad_target,
    )

    # Natural Gradient Update: X = η - F⁻¹ * ∇_η E
    X .= η .- inv_fisher * grad_eta

    return X
end

# Helper to create state
struct ClosedFormStrategyState{T}
    target::T
end

function ExponentialFamilyProjection.create_state!(
    strategy::ClosedFormStrategy,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    initial_ef,
    supplementary_η,
)
    return ClosedFormStrategyState(projection_argument)
end

function ExponentialFamilyProjection.prepare_state!(
    strategy::ClosedFormStrategy,
    state::ClosedFormStrategyState,
    M::AbstractManifold,
    parameters::ProjectionParameters,
    projection_argument,
    current_ef,
    supplementary_η,
)
    return state
end

# Compute cost for logging/convergence check
function ExponentialFamilyProjection.compute_cost(
    M::AbstractManifold,
    strategy::ClosedFormStrategy,
    state::ClosedFormStrategyState,
    η,
    logpartition,
    gradlogpartition,
    inv_fisher,
)
    # Cost = KL(q || p) = E_q[log q] - E_q[log p]

    # Reconstruct distribution
    q_dist = convert(
        ExponentialFamilyDistribution,
        M,
        ExponentialFamilyManifolds.partition_point(M, η),
    )
    dist_std = convert(Distribution, q_dist)

    # E_q[log p] via CFE
    E_log_p = mean(ClosedFormExpectation(), state.target, dist_std)

    # E_q[log q] = -entropy(q)
    return -entropy(dist_std) - E_log_p
end

# preprocess_strategy_argument for ClosedFormStrategy
# Special handling for RxInfer closures that wrap ProductOf
function ExponentialFamilyProjection.preprocess_strategy_argument(
    strategy::ClosedFormStrategy,
    argument::Function,
)
    # RxInfer wraps ProductOf in a closure. 
    # Extract the ProductOf from the closure's captured variables.
    # The closure typically has one field holding the ProductOf.
    fn_type = typeof(argument)
    field_names = fieldnames(fn_type)

    if !isempty(field_names)
        # Get the first field (usually the captured ProductOf)
        captured = getfield(argument, first(field_names))

        # If it's a ProductOf, use it directly
        if captured isa ProductOf
            return (strategy, Logpdf(captured))
        end

        # If it's a Distribution (e.g. LogNormal inside ProjectionExt closure), use it directly
        if captured isa Distribution
            return (strategy, Logpdf(captured))
        end
    end

    # Fallback: keep the function as-is
    return (strategy, argument)
end

# Generic fallback for non-Function arguments
function ExponentialFamilyProjection.preprocess_strategy_argument(
    strategy::ClosedFormStrategy,
    argument::Distribution,
)
    # ClosedFormStrategy accepts any callable or distribution as argument
    return (strategy, Logpdf(argument))
end

# Generic fallback for non-Function arguments
function ExponentialFamilyProjection.preprocess_strategy_argument(
    strategy::ClosedFormStrategy,
    argument,
)
    # ClosedFormStrategy accepts any callable or distribution as argument
    return (strategy, argument)
end

end
