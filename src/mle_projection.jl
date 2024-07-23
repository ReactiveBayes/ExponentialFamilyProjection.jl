export mle_projection
using ForwardDiff

function targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    return -mean(logpdf(ef, data))
end

function grad_targetfn(M, p, data)
    ef = convert(ExponentialFamilyDistribution, M, p)
    invfisher = cholinv(Hermitian(fisherinformation(ef)))
    X = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, invfisher*ForwardDiff.gradient((p) -> targetfn(M, p, data),p))
    return ExponentialFamilyProjection.Manopt.project(M, p, X)
end

# function grad_targetfn(M, p, data)
#     ef = convert(ExponentialFamilyDistribution, M, p)
#     invfisher = cholinv(Hermitian(fisherinformation(ef)))
#     suffstats = map(sample -> sufficientstatistics(ef, sample), data)

#     # X = ExponentialFamilyProjection.ExponentialFamilyManifolds.partition_point(M, invfisher*ForwardDiff.gradient((p) -> targetfn(M, p, data),p))
#     # return ExponentialFamilyProjection.Manopt.project(M, p, X)
# end


function mle_projection(
    prj::ProjectedTo,
    data;
    initialpoint = nothing,
    kwargs...,
)
    M          = get_projected_to_manifold(prj)
    parameters = get_projected_to_parameters(prj)
    f          = (m,p) -> targetfn(m, p, data)
    g          = (m,p) -> grad_targetfn(m, p, data)
   

    initialpoint = preprocess_initialpoint(initialpoint, M, parameters)

    kwargs = !haskey(kwargs, :debug) ? (; kwargs..., debug = missing) : kwargs

    q = gradient_descent(
            M,
            f,
            g,
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