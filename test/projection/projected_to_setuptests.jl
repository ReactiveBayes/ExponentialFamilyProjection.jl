using ExponentialFamily, Distributions, BayesBase, StableRNGs, RollingFunctions, Manopt

function test_projection_convergence(distribution; to = missing, dims = missing, kwargs...)
    targetfn = let distribution = distribution
        (x) -> logpdf(distribution, x)
    end

    ef = convert(ExponentialFamilyDistribution, distribution)
    T = ismissing(to) ? ExponentialFamily.exponential_family_typetag(ef) : to
    dims = ismissing(dims) ? size(rand(StableRNG(42), distribution)) : dims

    test1 = test_convergence_nsamples(distribution, targetfn, ef, T, dims; kwargs...)

    if !test1
        @warn "`nsamples` convergence test for $(distribution) failed."
        return false
    end

    test2 = test_convergence_niterations(distribution, targetfn, ef, T, dims; kwargs...)

    if !test2
        @warn "`niterations` convergence test for $(distribution) failed."
        return false
    end

    return test1 && test2
end

_convergence_nsamples_default_range(distribution) =
    _convergence_nsamples_default_range(variate_form(typeof(distribution)), distribution)

_convergence_nsamples_default_range(::Type{Univariate}, distribution) = 100:50:1000
_convergence_nsamples_default_range(::Type{Multivariate}, distribution) = 500:100:2000

_convergence_nsamples_default_tolerance(distribution) =
    _convergence_nsamples_default_tolerance(
        variate_form(typeof(distribution)),
        distribution,
    )

_convergence_nsamples_default_tolerance(::Type{Univariate}, distribution) = 1e-6
_convergence_nsamples_default_tolerance(::Type{Multivariate}, distribution) = 1e-6

_convergence_nsamples_default_niterations(distribution) =
    _convergence_nsamples_default_niterations(
        variate_form(typeof(distribution)),
        distribution,
    )

_convergence_nsamples_default_niterations(::Type{Univariate}, distribution) = 500
_convergence_nsamples_default_niterations(::Type{Multivariate}, distribution) = 1_000

function test_convergence_nsamples(
    distribution,
    targetfn,
    ef,
    T,
    dims;
    nsamples_range = _convergence_nsamples_default_range(distribution),
    nsamples_tolerance = _convergence_nsamples_default_tolerance(distribution),
    nsamples_niterations = _convergence_nsamples_default_niterations(distribution),
    nsamples_rng = StableRNG(42),
    nsamples_stepsize = ConstantStepsize(0.1),
    kwargs...,
)
    divergence = map(nsamples_range) do nsamples
        parameters = ProjectionParameters(
            nsamples = nsamples,
            niterations = nsamples_niterations,
            tolerance = nsamples_tolerance,
            seed = rand(nsamples_rng, UInt),
            stepsize = nsamples_stepsize,
        )
        projection = ProjectedTo(
            T,
            dims...,
            parameters = parameters,
            conditioner = getconditioner(ef),
        )
        approximated = project_to(projection, targetfn)
        return kldivergence(approximated, distribution)
    end

    return test_convergence_to_stable_point(divergence)
end

_convergence_niterations_default_range(distribution) =
    _convergence_niterations_default_range(variate_form(typeof(distribution)), distribution)

_convergence_niterations_default_range(::Type{Univariate}, distribution) = 100:50:1000
_convergence_niterations_default_range(::Type{Multivariate}, distribution) = 100:50:1000

_convergence_niterations_default_tolerance(distribution) =
    _convergence_niterations_default_tolerance(
        variate_form(typeof(distribution)),
        distribution,
    )

_convergence_niterations_default_tolerance(::Type{Univariate}, distribution) = missing
_convergence_niterations_default_tolerance(::Type{Multivariate}, distribution) = missing

_convergence_niterations_default_nsamples(distribution) =
    _convergence_niterations_default_nsamples(
        variate_form(typeof(distribution)),
        distribution,
    )

_convergence_niterations_default_nsamples(::Type{Univariate}, distribution) = 500
_convergence_niterations_default_nsamples(::Type{Multivariate}, distribution) = 1_000

function test_convergence_niterations(
    distribution,
    targetfn,
    ef,
    T,
    dims;
    niterations_range = _convergence_niterations_default_range(distribution),
    niterations_tolerance = _convergence_niterations_default_tolerance(distribution),
    niterations_nsamples = _convergence_niterations_default_nsamples(distribution),
    niterations_rng = StableRNG(42),
    niterations_stepsize = ConstantStepsize(0.1),
    kwargs...,
)
    divergence = map(niterations_range) do niterations
        parameters = ProjectionParameters(
            nsamples = niterations_nsamples,
            niterations = niterations,
            tolerance = niterations_tolerance,
            seed = rand(niterations_rng, UInt),
            stepsize = niterations_stepsize,
        )
        projection = ProjectedTo(
            T,
            dims...,
            parameters = parameters,
            conditioner = getconditioner(ef),
        )
        approximated = project_to(projection, targetfn)
        return kldivergence(approximated, distribution)
    end

    return test_convergence_to_stable_point(divergence)
end

function test_convergence_to_stable_point(
    series;
    window_size = (5, 10),
    stdthreshold = 5e-2,
    valthreshold = 1e-4,
)
    if all(<(valthreshold), abs.(series))
        return true
    end

    # We check for each `window_size` that the moving std converges to zero
    for ws in window_size
        movingstd = rolling(std, series, min(ws, length(series)))

        # We try to find number of consecutive std point 
        # which are less than `stdthreshold`
        count = 0
        for i in eachindex(movingstd)
            if movingstd[i] < stdthreshold
                count += 1
            else
                count = 0
            end
        end

        # For the test to pass we require that at least `20%` of
        # consecutive points at the end are less than `stdthreshold`
        if count < (length(movingstd) ÷ 5)
            return false
        end

    end

    return true
end