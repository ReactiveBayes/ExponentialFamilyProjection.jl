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

Base.@kwdef struct ProjectionParameters{I,S,T,P,E}
    niterations::I = 100
    nsamples::S = 2000
    tolerance::T = 1e-6
    stepsize::P = ConstantStepsize(0.1)
    seed::E = 42
end

const DefaultProjectionParameters = ProjectionParameters()

getniterations(parameters::ProjectionParameters) = parameters.niterations
getnsamples(parameters::ProjectionParameters) = parameters.nsamples
gettolerance(parameters::ProjectionParameters) = parameters.tolerance
getstepsize(parameters::ProjectionParameters) = parameters.stepsize
getseed(parameters::ProjectionParameters) = parameters.seed

function Manopt.get_stopping_criterion(parameters::ProjectionParameters)
    stopping = StopAfterIteration(getniterations(parameters))

    if !ismissing(gettolerance(parameters)) && !isnothing(gettolerance(parameters))
        stopping = stopping | StopWhenGradientNormLess(gettolerance(parameters))
    end

    return stopping
end

using Manopt, StaticTools

export project_to

function project_to(prj::ProjectedTo, f::F) where {F}
    parameters = get_projected_to_parameters(prj)
    M = get_projected_to_manifold(prj)
    seed = getseed(parameters)
    srng = StableRNG(seed)
    p0 = rand(srng, M)

    nsamples = getnsamples(parameters)
    samples = rand(srng, convert(ExponentialFamilyDistribution, M, p0), nsamples)
    logpdfs = zeros(eltype(p0), nsamples)
    sufficientstatistics = zeros(eltype(p0), length(p0), nsamples)
    gradsamples = similar(sufficientstatistics)

    buffer = MallocSlabBuffer()
    try
        state = ControlVariateStrategyState(
            samples = samples,
            logpdfs = logpdfs,
            sufficientstatistics = sufficientstatistics,
            gradsamples = gradsamples,
        )
        strategy = ControlVariateStrategy(
            nsamples = nsamples,
            seed = seed,
            rng = srng,
            state = state,
            buffer = buffer,
        )

        inplacef = convert(InplaceLogpdf, f)
        g_grad_g! = CVICostGradientObjective(inplacef, strategy)
        objective =
            ManifoldCostGradientObjective(g_grad_g!; evaluation = InplaceEvaluation())

        q = gradient_descent!(
            M,
            objective,
            p0;
            stopping_criterion = get_stopping_criterion(parameters),
            stepsize = getstepsize(parameters),
            debug = missing,
        )

        return convert(
            get_projected_to_type(prj),
            convert(ExponentialFamilyDistribution, M, q),
        )
    catch e
        rethrow(e)
    finally
        free(buffer)
    end
end

