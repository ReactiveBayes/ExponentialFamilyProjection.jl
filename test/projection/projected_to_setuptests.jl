using ExponentialFamily, Distributions, BayesBase, StableRNGs, RollingFunctions, Manopt

function test_projection_mle(
    distribution;
    to = missing,
    dims = missing,
    conditioner = missing,
    kwargs...,
)
 
    T =
        ismissing(to) ?
        ExponentialFamily.exponential_family_typetag(
            convert(ExponentialFamilyDistribution, distribution),
        ) : to
    dims = ismissing(dims) ? size(rand(StableRNG(42), distribution)) : dims
    conditioner =
        ismissing(conditioner) ?
        getconditioner(convert(ExponentialFamilyDistribution, distribution)) : conditioner

    test1 =
        test_convergence_nsamples_mle(distribution, T, dims, conditioner; kwargs...)
    test2 = test_convergence_niterations_mle(
        distribution,
        T,
        dims,
        conditioner;
        kwargs...,
    )

    return test1 && test2
end

function test_projection_convergence(
    distribution;
    to = missing,
    dims = missing,
    conditioner = missing,
    kwargs...,
)
    targetfn = let distribution = distribution
        (x) -> logpdf(distribution, x)
    end

    T =
        ismissing(to) ?
        ExponentialFamily.exponential_family_typetag(
            convert(ExponentialFamilyDistribution, distribution),
        ) : to
    dims = ismissing(dims) ? size(rand(StableRNG(42), distribution)) : dims
    conditioner =
        ismissing(conditioner) ?
        getconditioner(convert(ExponentialFamilyDistribution, distribution)) : conditioner

    test1 =
        test_convergence_nsamples(distribution, targetfn, T, dims, conditioner; kwargs...)
    test2 = test_convergence_niterations(
        distribution,
        targetfn,
        T,
        dims,
        conditioner;
        kwargs...,
    )

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
    T,
    dims,
    conditioner;
    nsamples_range = _convergence_nsamples_default_range(distribution),
    nsamples_tolerance = _convergence_nsamples_default_tolerance(distribution),
    nsamples_niterations = _convergence_nsamples_default_niterations(distribution),
    nsamples_rng = StableRNG(42),
    nsamples_stepsize = ConstantStepsize(0.1),
    nsamples_required_accuracy = 1e-1,
    kwargs...,
)

    experiment = map(nsamples_range) do nsamples
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.ControlVariateStrategy(
                nsamples = nsamples,
                seed = rand(nsamples_rng, UInt),
            ),
            niterations = nsamples_niterations,
            tolerance = nsamples_tolerance,
            stepsize = nsamples_stepsize,
        )
        projection =
            ProjectedTo(T, dims..., parameters = parameters, conditioner = conditioner)
        approximated = project_to(projection, targetfn)
        divergence = test_convergence_metric(approximated, distribution)
        return divergence, approximated
    end

    divergence = map(e -> e[1], experiment)
    approximated = map(e -> e[2], experiment)

    test_required_accuracy = any(<(nsamples_required_accuracy), divergence)

    if !test_required_accuracy
        @warn "`nsamples` accuracy test for `$(distribution)` failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
    end

    test_convergence = test_convergence_to_stable_point(divergence)

    if !test_convergence
        @warn "`nsamples` convergence test for $(distribution) failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
        return false
    end

    return test_required_accuracy && test_convergence
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
    T,
    dims,
    conditioner;
    niterations_range = _convergence_niterations_default_range(distribution),
    niterations_tolerance = _convergence_niterations_default_tolerance(distribution),
    niterations_nsamples = _convergence_niterations_default_nsamples(distribution),
    niterations_rng = StableRNG(42),
    niterations_stepsize = ConstantStepsize(0.1),
    niterations_required_accuracy = 1e-1,
    kwargs...,
)

    experiment = map(niterations_range) do niterations
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.ControlVariateStrategy(
                nsamples = niterations_nsamples,
                seed = rand(niterations_rng, UInt),
            ),
            niterations = niterations,
            tolerance = niterations_tolerance,
            stepsize = niterations_stepsize,
        )
        projection =
            ProjectedTo(T, dims..., parameters = parameters, conditioner = conditioner)
        approximated = project_to(projection, targetfn)
        divergence = test_convergence_metric(approximated, distribution)
        return divergence, approximated
    end

    divergence = map(e -> e[1], experiment)
    approximated = map(e -> e[2], experiment)

    test_required_accuracy = any(<(niterations_required_accuracy), divergence)

    if !test_required_accuracy
        @warn "`niterations` accuracy test for `$(distribution)` failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
    end

    test_convergence = test_convergence_to_stable_point(divergence)

    if !test_convergence
        @warn "`niterations` convergence test for $(distribution) failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
        return false
    end

    return test_required_accuracy && test_convergence
end


function test_convergence_niterations_mle(
    distribution,
    T,
    dims,
    conditioner;
    niterations_range = _convergence_niterations_default_range(distribution),
    niterations_tolerance = _convergence_niterations_default_tolerance(distribution),
    niterations_nsamples = _convergence_niterations_default_nsamples(distribution),
    niterations_rng = StableRNG(42),
    niterations_stepsize = ConstantStepsize(0.1),
    niterations_required_accuracy = 1e-1,
    number_data_points = 1000,
    kwargs...,
)
    data = rand(niterations_rng, distribution, number_data_points)
    experiment = map(niterations_range) do niterations
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.ControlVariateStrategy(
                nsamples = niterations_nsamples,
                seed = rand(niterations_rng, UInt),
            ),
            niterations = niterations,
            tolerance = niterations_tolerance,
            stepsize = niterations_stepsize,
        )
        projection =
            ProjectedTo(T, dims..., parameters = parameters, conditioner = conditioner)
        approximated = mle_projection(projection, data)
        divergence = test_convergence_metric(approximated, distribution)
        return divergence, approximated
    end

    divergence = map(e -> e[1], experiment)
    approximated = map(e -> e[2], experiment)

    test_required_accuracy = any(<(niterations_required_accuracy), divergence)

    if !test_required_accuracy
        @warn "`niterations` accuracy test for `$(distribution)` failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
    end

    test_convergence = test_convergence_to_stable_point(divergence)

    if !test_convergence
        @warn "`niterations` convergence test for $(distribution) failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
        return false
    end

    return test_required_accuracy && test_convergence
end


function test_convergence_nsamples_mle(
    distribution,
    T,
    dims,
    conditioner;
    nsamples_range = _convergence_nsamples_default_range(distribution),
    nsamples_tolerance = _convergence_nsamples_default_tolerance(distribution),
    nsamples_niterations = _convergence_nsamples_default_niterations(distribution),
    nsamples_rng = StableRNG(42),
    nsamples_stepsize = ConstantStepsize(0.1),
    nsamples_required_accuracy = 1e-1,
    number_data_points = 1000,
    kwargs...,
)
    
    data = rand(nsamples_rng, distribution, number_data_points)
    experiment = map(nsamples_range) do nsamples
        parameters = ProjectionParameters(
            strategy = ExponentialFamilyProjection.ControlVariateStrategy(
                nsamples = nsamples,
                seed = rand(nsamples_rng, UInt),
            ),
            niterations = nsamples_niterations,
            tolerance = nsamples_tolerance,
            stepsize = nsamples_stepsize,
        )
        
        projection =
            ProjectedTo(T, dims..., parameters = parameters, conditioner = conditioner)
        approximated = mle_projection(projection, data)
        divergence = test_convergence_metric(approximated, distribution)
        return divergence, approximated
    end

    divergence = map(e -> e[1], experiment)
    approximated = map(e -> e[2], experiment)

    test_required_accuracy = any(<(nsamples_required_accuracy), divergence)

    if !test_required_accuracy
        @warn "`nsamples` accuracy test for `$(distribution)` failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
    end

    test_convergence = test_convergence_to_stable_point(divergence)

    if !test_convergence
        @warn "`nsamples` convergence test for $(distribution) failed. The approximated distributions were `$(approximated)`. The divergences was `$(divergence)`."
        return false
    end

    return test_required_accuracy && test_convergence
end



# The metric we are using in the tests is `KL` divergence
function test_convergence_metric(left, right)
    return kldivergence(left, right)
end

# For the `ProductOf` we assume that we can call the `prod` function 
# to compute the result. That means that we compare against analytical 
# solutions for the `ProductOf` distributions.
function test_convergence_metric(left, right::ProductOf)
    return kldivergence(
        left,
        prod(
            PreserveTypeProd(Distribution),
            BayesBase.getleft(right),
            BayesBase.getright(right),
        ),
    )
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

