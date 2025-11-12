"""
    DefaultStrategy

The DefaultStrategy selects the optimal projection strategy based on the type of the second argument provided to the `project_to` function.

Rules:
- If the second argument is an `AbstractArray`, use `MLEStrategy`.
- For all other types, use `ControlVariateStrategy`.

!!! note
    The rules above are subject to change.
"""
struct DefaultStrategy end

preprocess_strategy_argument(::DefaultStrategy, argument::AbstractArray) =
    preprocess_strategy_argument(MLEStrategy(), argument)
preprocess_strategy_argument(::DefaultStrategy, argument::Any) =
    preprocess_strategy_argument(ControlVariateStrategy(), argument)
