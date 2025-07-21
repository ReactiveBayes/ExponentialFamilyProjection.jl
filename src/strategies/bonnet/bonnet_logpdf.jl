
struct BonnetInplaceLogpdf{F,G,H}
    logpdf!::F
    grad!::G
    hess!::H
end

"""
    InplaceLogpdfGradHess(logpdf!, grad!, hess!)

Wraps `logpdf!`, `grad!`, and `hess!` functions in a type that can be used for dispatch.
The structure allows for separate evaluation of log probability density, gradient, and Hessian.

# Arguments
- `logpdf!`: Function that takes `(out, x)` and writes the log probability density to `out`
- `grad!`: Function that takes `(out, x)` and writes the gradient to `out`  
- `hess!`: Function that takes `(out, x)` and writes the Hessian to `out`

# Methods
- `logpdf!(structure, out, x)`: Evaluate log probability density
- `grad!(structure, out, x)`: Evaluate gradient
- `hess!(structure, out, x)`: Evaluate Hessian

All methods expect pre-allocated containers `out` of appropriate dimensions.
"""
struct InplaceLogpdfGradHess{F,G,H}
    logpdf!::F
    grad!::G
    hess!::H
end

"""
    logpdf!(inplace::InplaceLogpdfGradHess, out, x)

Evaluate the log probability density function at point `x`, writing the result to pre-allocated container `out`.
"""
function logpdf!(inplace::InplaceLogpdfGradHess, out, x)
    inplace.logpdf!(out, x)
    return out
end

"""
    grad!(inplace::InplaceLogpdfGradHess, out, x)

Evaluate the gradient at point `x`, writing the result to pre-allocated container `out`.
"""
function grad!(inplace::InplaceLogpdfGradHess, out, x)
    inplace.grad!(out, x)
    return out
end

"""
    hess!(inplace::InplaceLogpdfGradHess, out, x)

Evaluate the Hessian at point `x`, writing the result to pre-allocated container `out`.
"""
function hess!(inplace::InplaceLogpdfGradHess, out, x)
    inplace.hess!(out, x)
    return out
end

function Base.convert(::Type{InplaceLogpdfGradHess}, something::InplaceLogpdfGradHess)
    return something
end
