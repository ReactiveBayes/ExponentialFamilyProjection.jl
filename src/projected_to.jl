using ExponentialFamily, Manopt

export ProjectedTo, ProjectionParameters

struct ProjectedTo{T,D,C,P}
    dims::D
    conditioner::C
    parameters::P
end

ProjectedTo(
    dims::Vararg{Int};
    conditioner = nothing,
    parameters = DefaultProjectionParameters,
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
    parameters::P = DefaultProjectionParameters,
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

Base.@kwdef struct ProjectionParameters{I,S,T,P,E,B}
    niterations::I = 100
    nsamples::S = 2000
    tolerance::T = 1e-6
    stepsize::P = ConstantStepsize(0.1)
    seed::E = 42
    usebuffer::B = Val(true)
end

const DefaultProjectionParameters = ProjectionParameters()

getniterations(parameters::ProjectionParameters) = parameters.niterations
getnsamples(parameters::ProjectionParameters) = parameters.nsamples
gettolerance(parameters::ProjectionParameters) = parameters.tolerance
getstepsize(parameters::ProjectionParameters) = parameters.stepsize
getseed(parameters::ProjectionParameters) = parameters.seed

with_buffer(f::F, parameters::ProjectionParameters) where {F} =
    with_buffer(f, parameters.usebuffer, parameters)

with_buffer(f::F, ::Val{false}, parameters::ProjectionParameters) where {F} = f(nothing)
with_buffer(f::F, ::Val{true}, parameters::ProjectionParameters) where {F} =
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
    stopping = StopAfterIteration(getniterations(parameters))

    if !ismissing(gettolerance(parameters)) && !isnothing(gettolerance(parameters))
        stopping = stopping | StopWhenGradientNormLess(gettolerance(parameters))
    end

    return stopping
end

using Manopt, StaticTools

export project_to

function project_to(prj::ProjectedTo, f::F, supplementary...) where {F}
    parameters = get_projected_to_parameters(prj)
    M = get_projected_to_manifold(prj)
    seed = getseed(parameters)
    rng = StableRNG(seed)
    initialpoint = rand(rng, M)

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
        return getnaturalparameters(supplementary_ef)
    end

    nsamples = getnsamples(parameters)
    samples = rand(rng, convert(ExponentialFamilyDistribution, M, initialpoint), nsamples)
    logpdfs = zeros(eltype(initialpoint), nsamples)
    sufficientstatistics = zeros(eltype(initialpoint), length(initialpoint), nsamples)
    gradsamples = similar(sufficientstatistics)

    return with_buffer(parameters) do buffer
        state = ControlVariateStrategyState(
            samples = samples,
            logpdfs = logpdfs,
            sufficientstatistics = sufficientstatistics,
            gradsamples = gradsamples,
        )
        strategy = ControlVariateStrategy(
            nsamples = nsamples,
            seed = seed,
            rng = rng,
            state = state,
        )

        g_grad_g! = CVICostGradientObjective(f, supplementary_η, strategy, buffer)
        objective =
            ManifoldCostGradientObjective(g_grad_g!; evaluation = InplaceEvaluation())

        q = gradient_descent!(
            M,
            objective,
            initialpoint;
            stopping_criterion = get_stopping_criterion(parameters),
            stepsize = getstepsize(parameters),
            debug = missing,
        )

        return convert(
            get_projected_to_type(prj),
            convert(ExponentialFamilyDistribution, M, q),
        )
    end
end

