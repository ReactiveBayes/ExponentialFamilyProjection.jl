using ExponentialFamily, Manopt

export ProjectedTo, ProjectionParameters, project_to

"""
    ProjectedTo(::Type{T}, dims...; conditioner = nothing, parameters = DefaultProjectionParameters)

A specification of a projection to an exponential family distribution.

The following arguments are required:

* `Type{T}`: a type of an exponential family member to project to, e.g. `Beta`
* `dims...`: dimensions of the distribution, e.g. `2` for `MvNormal`

The following arguments are optional:

* `conditioner = nothing`: a conditioner to use for the projection, not all exponential family members require a conditioner, but some do, e.g. `Laplace`
* `parameters = DefaultProjectionParameters`: parameters for the projection procedure
* `kwargs = nothing`: Additional arguments passed to `Manopt.gradient_descent!` (optional). For details on `gradient_descent!` parameters, see the [Manopt.jl documentation](https://manoptjl.org/stable/solvers/gradient_descent/#Manopt.gradient_descent). Note, that `kwargs` passed to `project_to` take precedence over `kwargs` specified in the parameters.

```jldoctest 
julia> using ExponentialFamily

julia> projected_to = ProjectedTo(Beta)
ProjectedTo(Beta)

julia> projected_to = ProjectedTo(Beta, parameters = ProjectionParameters(niterations = 10))
ProjectedTo(Beta)

julia> projected_to = ProjectedTo(MvNormalMeanCovariance, 2)
ProjectedTo(MvNormalMeanCovariance, dims = 2)

julia> projected_to = ProjectedTo(Laplace, conditioner = 2.0)
ProjectedTo(Laplace, conditioner = 2.0)
```
"""
struct ProjectedTo{T,D,C,P,E}
    dims::D
    conditioner::C
    parameters::P
    kwargs::E
end

ProjectedTo(
    dims::Tuple{Vararg{Int}};
    conditioner = nothing,
    parameters = DefaultProjectionParameters(),
    kwargs = nothing,
) = ProjectedTo(
    ExponentialFamilyDistribution,
    dims...,
    conditioner = conditioner,
    parameters = parameters,
    kwargs = kwargs,
)
ProjectedTo(;
    conditioner = nothing,
    parameters = DefaultProjectionParameters(),
    kwargs = nothing,
) = ProjectedTo(
    ExponentialFamilyDistribution,
    ()...,
    conditioner = conditioner,
    parameters = parameters,
    kwargs = kwargs,
)
ProjectedTo(
    dim::Int;
    conditioner = nothing,
    parameters = DefaultProjectionParameters(),
    kwargs = nothing,
) = ProjectedTo(
    ExponentialFamilyDistribution,
    dim,
    conditioner = conditioner,
    parameters = parameters,
    kwargs = kwargs,
)
function ProjectedTo(
    ::Type{T},
    dims...;
    conditioner::C = nothing,
    parameters::P = DefaultProjectionParameters(),
    kwargs::E = nothing,
) where {T,C,P,E}
    # Check that `dims` are all integers
    if !all(d -> typeof(d) <: Int, dims)
        # If not, throw an error, also suggesting to use keyword arguments
        msg =
            lazy"The dimensions must be integers, but `$(typeof(dims))` has been provided. Use `conditioner = ...` keyword argument to supply conditioner."
        if any(d -> typeof(d) <: ProjectionParameters, dims)
            msg = lazy"$(msg) Use `parameters = ...` keyword argument to supply parameters."
        end
        error(msg)
    end
    return ProjectedTo{T,typeof(dims),C,P,E}(dims, conditioner, parameters, kwargs)
end

get_projected_to_type(::ProjectedTo{T}) where {T} = T
get_projected_to_dims(prj::ProjectedTo) = prj.dims
get_projected_to_conditioner(prj::ProjectedTo) = prj.conditioner
get_projected_to_parameters(prj::ProjectedTo) = prj.parameters
get_projected_to_kwargs(prj::ProjectedTo) = prj.kwargs
get_projected_to_manifold(prj::ProjectedTo) =
    ExponentialFamilyManifolds.get_natural_manifold(
        get_projected_to_type(prj),
        get_projected_to_dims(prj),
        get_projected_to_conditioner(prj),
    )

function Base.show(io::IO, prj::ProjectedTo)
    print(io, "ProjectedTo(", get_projected_to_type(prj))
    dims = get_projected_to_dims(prj)
    if !isempty(dims)
        print(io, ", dims = ")
        join(io, dims, ", ")
    end
    conditioner = get_projected_to_conditioner(prj)
    if !isnothing(conditioner)
        print(io, ", conditioner = ", conditioner)
    end
    print(io, ")")
end

"""
    ProjectionParameters(; kwargs...)

A type to hold the parameters for the projection procedure. 
The following parameters are available:
    
* `strategy = ExponentialFamilyProjection.DefaultStrategy()`: The strategy to use to compute the gradients.
* `niterations = 100`: The number of iterations for the optimization procedure.
* `tolerance = 1e-6`: The tolerance for the norm of the gradient.
* `stepsize = ConstantLength(0.1)`: The stepsize for the optimization procedure. Accepts stepsizes from `Manopt.jl`.
* `seed`: Optional; Seed for the `rng`
* `rng`: Optional; Random number generator
* `direction = BoundedNormUpdateRule(static(1.0)`: Direction update rule. Accepts `Manopt.DirectionUpdateRule` from `Manopt.jl`.
"""
Base.@kwdef struct ProjectionParameters{S,I,T,P,D,N,U}
    strategy::S = DefaultStrategy()
    niterations::I = 100
    tolerance::T = 1e-6
    stepsize::P = ConstantLength(0.1)
    seed::D = 42
    rng::N = StableRNG(seed)
    direction::U = BoundedNormUpdateRule(static(1.0))
end

"""
    DefaultProjectionParameters()

Return the default parameters for the projection procedure.
"""
DefaultProjectionParameters() = ProjectionParameters() # do not use `const DefaultProjectionParameters = ProjectionParameters()` here since it reuses the `rng` then

getstrategy(parameters::ProjectionParameters) = parameters.strategy
getniterations(parameters::ProjectionParameters) = parameters.niterations
gettolerance(parameters::ProjectionParameters) = parameters.tolerance
getstepsize(parameters::ProjectionParameters) = parameters.stepsize
getseed(parameters::ProjectionParameters) = parameters.seed
getrng(parameters::ProjectionParameters) = parameters.rng
getdirection(parameters::ProjectionParameters) = parameters.direction

"""
    getinitialpoint(strategy, M::AbstractManifold, parameters::ProjectionParameters)

Returns an initial point to start optimization from. By default returns a `rand` point from `M`, 
but different strategies may implement their own methods.
"""
function getinitialpoint(::Any, M::AbstractManifold, parameters::ProjectionParameters)
    return rand(getrng(parameters), M)
end

function Manopt.get_stopping_criterion(parameters::ProjectionParameters)
    return Manopt.get_stopping_criterion(
        parameters::ProjectionParameters,
        getniterations(parameters),
        gettolerance(parameters),
    )
end

function Manopt.get_stopping_criterion(
    parameters::ProjectionParameters,
    niterations,
    tolerance,
)
    return StopAfterIteration(niterations) | StopWhenGradientNormLess(tolerance)
end

function Manopt.get_stopping_criterion(
    parameters::ProjectionParameters,
    niterations::Missing,
    tolerance,
)
    return StopWhenGradientNormLess(tolerance)
end

function Manopt.get_stopping_criterion(
    parameters::ProjectionParameters,
    niterations,
    tolerance::Missing,
)
    return StopAfterIteration(niterations)
end

using Manopt

function check_inputs(prj::ProjectedTo, projection_argument::F, supplementary...; initialpoint = nothing, kwargs...) where {F}
    if isnothing(initialpoint)
        return
    end
    if !(initialpoint ∈ get_projected_to_manifold(prj))
        return error(
            lazy"The initial point must be on the manifold `$(get_projected_to_manifold(prj))`, got `$(typeof(initialpoint))`",
        )
    end
end 
"""
    project_to(to::ProjectedTo, argument::F, supplementary..., initialpoint, kwargs...)

Finds the closest projection of `argument` onto the exponential family distribution specified by `to`.

# Arguments
- `to::ProjectedTo`: Configuration for the projection. Refer to `ProjectedTo` for detailed information.
- `argument::F`: An (un-normalized) function representing the log-PDF of an arbitrary distribution _or_ a list of samples.
- `supplementary...`: Additional distributions to project the product of `argument` and these distributions (optional).
- `initialpoint`: Starting point for the optimization process (optional).
- `kwargs...`: Additional arguments passed to `Manopt.gradient_descent!` (optional). For details on `gradient_descent!` parameters, see the [Manopt.jl documentation](https://manoptjl.org/stable/solvers/gradient_descent/#Manopt.gradient_descent).

# Supplementary

The `supplementary` distributions must match the type and conditioner of the target distribution specified in `to`. 
Including supplementary distributions is equivalent to modified `argument` function as follows:

```julia
f_modified = (x) -> argument(x) + logpdf(supplementary[1], x) + logpdf(supplementary[2], x) + ...
```

```jldoctest
julia> using ExponentialFamily, BayesBase

julia> f = (x) -> logpdf(Beta(30.14, 2.71), x);

julia> prj = ProjectedTo(Beta; parameters = ProjectionParameters(niterations = 500))
ProjectedTo(Beta)

julia> project_to(prj, f) isa ExponentialFamily.Beta
true
```

```jldoctest
julia> using ExponentialFamily, BayesBase, StableRNGs

julia> samples = rand(StableRNG(42), Beta(30.14, 2.71), 1_000);

julia> prj = ProjectedTo(Beta; parameters = ProjectionParameters(tolerance = 1e-2))
ProjectedTo(Beta)

julia> project_to(prj, samples) isa ExponentialFamily.Beta
true
```

!!! note
    Different strategies are compatible with different types of arguments. Read optimization strategies section in the documentation for more information.
"""
function project_to(
    prj::ProjectedTo,
    projection_argument::F,
    supplementary...;
    initialpoint = nothing,
    kwargs...,
) where {F}
    M = get_projected_to_manifold(prj)
    projection_parameters = get_projected_to_parameters(prj)
    # "Supplementary" natural parameters are parameters that are simply being subtracted 
    # from the natural parameters of the current estiamted distribution. This might be useful 
    # to project a "product" of the function `f` and `supplementary` distributions
    supplementary_η = map(supplementary) do s
        if ExponentialFamily.exponential_family_typetag(s) !== get_projected_to_type(prj)
            error(
                lazy"Supplementary distributions must be of the same exponential member as the projection target `$(get_projected_to_type(prj))`, got `$(ExponentialFamily.exponential_family_typetag(s))`",
            )
        end
        supplementary_ef = convert(ExponentialFamilyDistribution, s)
        if getconditioner(supplementary_ef) !== get_projected_to_conditioner(prj)
            error(
                lazy"Supplementary distributions must have the same conditioner as the projection target `$(get_projected_to_type(prj))` with `conditioner = $(get_projected_to_conditioner(prj))`, got `$(ExponentialFamily.exponential_family_typetag(s))` with `conditioner = $(getconditioner(supplementary_ef))`",
            )
        end
        return copy(getnaturalparameters(supplementary_ef))
    end

    try
        projection_argument.logpdf!([0.0], randn(prj.dims))
    catch e
        error(
            "The supplied projection dimensions `$(prj.dims)` may be invalid for the provided logpdf! function. Check dimensions and logpdf! function.\n",
        )
    end

    strategy, projection_argument = preprocess_strategy_argument(
        getstrategy(projection_parameters),
        projection_argument,
    )
    current_iteration_point = preprocess_initialpoint(initialpoint, strategy, M, projection_parameters)
    check_inputs(prj, projection_argument, supplementary...; initialpoint = current_iteration_point, kwargs...)
    current_ef = convert(ExponentialFamilyDistribution, M, current_iteration_point)
    state = create_state!(
        strategy,
        M,
        projection_parameters,
        projection_argument,
        current_ef,
        supplementary_η,
    )

    # First we query the `kwargs` defined in the `ProjectionParameters`
    prj_kwargs = get_projected_to_kwargs(prj)
    prj_kwargs = isnothing(prj_kwargs) ? (;) : prj_kwargs
    # And attach the `kwargs` passed to `project_to`, those may override 
    # some settings in the `ProjectionParameters`
    if !isnothing(kwargs)
        prj_kwargs = (; prj_kwargs..., kwargs...)
    end
    # We disable the default `debug` statements, which are set in `Manopt` 
    # in order to improve the performance a little bit
    if !haskey(prj_kwargs, :debug)
        prj_kwargs = (; prj_kwargs..., debug = missing)
    end

    return _kernel_project_to(
        get_projected_to_type(prj),
        M,
        projection_parameters,
        projection_argument,
        supplementary_η,
        strategy,
        state,
        current_iteration_point,
        prj_kwargs,
    )
end

# see https://docs.julialang.org/en/v1/manual/performance-tips/#kernel-functions
# before this function call the argument may not be type-stable, inside these should be inferred properly
function _kernel_project_to(
    ::Type{T},
    M,
    projection_parameters,
    projection_argument,
    supplementary_η,
    strategy,
    state,
    current_iteration_point,
    kwargs,
) where {T}
    g_grad_g! = ProjectionCostGradientObjective(
        projection_parameters,
        projection_argument,
        copy(current_iteration_point),
        supplementary_η,
        strategy,
        state,
    )
    objective = ManifoldCostGradientObjective(g_grad_g!; evaluation = InplaceEvaluation())

    # `gradient_descent!` is a type-unstable call, so better not to use `q = gradient_descent!`
    # `gradient_descent!` will override `q` instead
    q = current_iteration_point
    direction = getdirection(projection_parameters)
    inited_direction = init_direction_rule(direction, M)
    gradient_descent!(
        M,
        objective,
        current_iteration_point;
        stopping_criterion = get_stopping_criterion(projection_parameters),
        stepsize = getstepsize(projection_parameters),
        direction = inited_direction,
        kwargs...,
    )

    return convert(T, convert(ExponentialFamilyDistribution, M, q))
end

# This function preprocess the initial point for the projection
# If the initial point is not provided, it generates a new one with the `getinitialpoint` function
function preprocess_initialpoint(initialpoint::Nothing, strategy, M, parameters)
    return getinitialpoint(strategy, M, parameters)
end

function preprocess_initialpoint(initialpoint::Any, strategy, M, parameters)
    return preprocess_initialpoint(
        ExponentialFamily.exponential_family_typetag(M),
        initialpoint,
        strategy,
        M,
        parameters,
    )
end

# If the initial point is provided as the distribution type which we project on to,
# we generate a new initial point using the `naturalparameters` of the distribution
function preprocess_initialpoint(
    ::Type{T},
    initialpoint::T,
    strategy,
    M,
    parameters,
) where {T}
    return preprocess_initialpoint(
        T,
        convert(ExponentialFamilyDistribution, initialpoint),
        strategy,
        M,
        parameters,
    )
end

function preprocess_initialpoint(
    ::Type{T},
    initialpoint::ExponentialFamilyDistribution{T},
    strategy,
    M,
    parameters,
) where {T}
    return ExponentialFamilyManifolds.partition_point(
        M,
        copy(getnaturalparameters(initialpoint)),
    )
end

# Otherwise we just copy the initial point, since we use it for the optimization in place
function preprocess_initialpoint(_, initialpoint::AbstractArray, strategy, M, parameters)
    return copy(initialpoint)
end