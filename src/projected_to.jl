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
struct ProjectedTo{T,D,C,P}
    dims::D
    conditioner::C
    parameters::P
end

ProjectedTo(
    dims::Vararg{Int};
    conditioner = nothing,
    parameters = DefaultProjectionParameters(),
) = ProjectedTo(
    ExponentialFamilyDistribution,
    dims...,
    conditioner = conditioner,
    parameters = parameters,
)
function ProjectedTo(
    ::Type{T},
    dims...;
    conditioner::C = nothing,
    parameters::P = DefaultProjectionParameters(),
) where {T,C,P}
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
    return ProjectedTo{T,typeof(dims),C,P}(dims, conditioner, parameters)
end

get_projected_to_type(::ProjectedTo{T}) where {T} = T
get_projected_to_dims(prj::ProjectedTo) = prj.dims
get_projected_to_conditioner(prj::ProjectedTo) = prj.conditioner
get_projected_to_parameters(prj::ProjectedTo) = prj.parameters
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
    
* `strategy = ExponentialFamilyProjection.ControlVariateStrategy()`: The strategy to use to compute the gradients.
* `niterations = 100`: The number of iterations for the optimization procedure.
* `tolerance = 1e-6`: The tolerance for the norm of the gradient.
* `stepsize = ConstantStepsize(0.1)`: The stepsize for the optimization procedure. Accepts stepsizes from `Manopt.jl`.
* `usebuffer = Val(true)`: Whether to use a buffer for the projection. Must be either `Val(true)` or `Val(false)`. Disabling buffer can be useful for debugging purposes.
"""
Base.@kwdef struct ProjectionParameters{S,I,T,P,B}
    strategy::S = ControlVariateStrategy()
    niterations::I = 100
    tolerance::T = 1e-6
    stepsize::P = ConstantStepsize(0.1)
    usebuffer::B = Val(true)
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

with_buffer(f::F, parameters::ProjectionParameters) where {F} =
    with_buffer(f, parameters.usebuffer, parameters)

with_buffer(f::F, buffer, ::ProjectionParameters) where {F} = f(buffer)
with_buffer(f::F, ::Val{false}, ::ProjectionParameters) where {F} = f(nothing)
with_buffer(f::F, ::Val{true}, ::ProjectionParameters) where {F} =
    let buffer = MallocSlabBuffer()
        try
            f(buffer)
        catch exception
            rethrow(exception)
        finally
            free(buffer)
        end
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

using Manopt, StaticTools

"""
    project_to(to::ProjectedTo, logf::F, supplementary..., initialpoint, kwargs...)

Finds the closest projection of `logf` onto the exponential family distribution specified by `to`.

# Arguments
- `to::ProjectedTo`: Configuration for the projection. Refer to `ProjectedTo` for detailed information.
- `logf::F`: An (un-normalized) function representing the log-PDF of an arbitrary distribution.
- `supplementary...`: Additional distributions to project the product of `logf` and these distributions (optional).
- `initialpoint`: Starting point for the optimization process (optional).
- `kwargs...`: Additional arguments passed to `Manopt.gradient_descent!` (optional). For details on `gradient_descent!` parameters, see the [Manopt.jl documentation](https://manoptjl.org/stable/solvers/gradient_descent/#Manopt.gradient_descent).

# Supplementary

The `supplementary` distributions must match the type and conditioner of the target distribution specified in `to`. 
Including supplementary distributions is equivalent to modified `logf` function as follows:

```julia
f_modified = (x) -> logf(x) + logpdf(supplementary[1], x) + logpdf(supplementary[2], x) + ...
```

```jldoctest
julia> using ExponentialFamily, BayesBase

julia> f = (x) -> logpdf(Beta(30.14, 2.71), x);

julia> prj = ProjectedTo(Beta; parameters = ProjectionParameters(niterations = 500))
ProjectedTo(Beta)

julia> project_to(prj, f) isa ExponentialFamily.Beta
true
```
"""
function project_to(
    prj::ProjectedTo,
    f::F,
    supplementary...;
    initialpoint = nothing,
    kwargs...,
) where {F}
    M = get_projected_to_manifold(prj)
    parameters = get_projected_to_parameters(prj)

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

    initialpoint = preprocess_initialpoint(initialpoint, M, parameters)

    state = prepare_state!(
        getstrategy(parameters),
        f,
        convert(ExponentialFamilyDistribution, M, initialpoint),
    )
    strategy = with_state(getstrategy(parameters), state)

    return with_buffer(parameters) do buffer

        g_grad_g! = CVICostGradientObjective(f, supplementary_η, strategy, buffer)
        objective =
            ManifoldCostGradientObjective(g_grad_g!; evaluation = InplaceEvaluation())

        q = gradient_descent!(
            M,
            objective,
            initialpoint;
            stopping_criterion = get_stopping_criterion(parameters),
            stepsize = getstepsize(parameters),
            direction = BoundedNormUpdateRule(static(1)),
            kwargs...,
        )

        return convert(
            get_projected_to_type(prj),
            convert(ExponentialFamilyDistribution, M, q),
        )
    end
end

# This function preprocess the initial point for the projection
# If the initial point is not provided, it generates a new one with the `getinitialpoint` function
function preprocess_initialpoint(initialpoint::Nothing, M, parameters)
    return getinitialpoint(getstrategy(parameters), M)
end

function preprocess_initialpoint(initialpoint::Any, M, parameters)
    return preprocess_initialpoint(
        ExponentialFamily.exponential_family_typetag(M),
        initialpoint,
        M,
        parameters,
    )
end

# If the initial point is provided as the distribution type which we project on to,
# we generate a new initial point using the `naturalparameters` of the distribution
function preprocess_initialpoint(::Type{T}, initialpoint::T, M, parameters) where {T}
    return preprocess_initialpoint(
        T,
        convert(ExponentialFamilyDistribution, initialpoint),
        M,
        parameters,
    )
end

function preprocess_initialpoint(
    ::Type{T},
    initialpoint::ExponentialFamilyDistribution{T},
    M,
    parameters,
) where {T}
    return ExponentialFamilyManifolds.partition_point(
        M,
        copy(getnaturalparameters(initialpoint)),
    )
end

# Otherwise we just copy the initial point, since we use it for the optimization in place
function preprocess_initialpoint(_, initialpoint::AbstractArray, M, parameters)
    return copy(initialpoint)
end