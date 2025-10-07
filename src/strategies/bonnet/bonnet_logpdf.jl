
"""
    InplaceLogpdfGradHess(logpdf!, grad_hess!)

Wraps `logpdf!` and the unified `grad_hess!` function in a type used for dispatch.
The unified interface evaluates gradient and Hessian together for efficiency.

# Arguments
- `logpdf!`: Function that takes `(out_logpdf, x)` and writes the logpdf to `out_logpdf`
- `grad_hess!`: Function that takes `(out_grad, out_hess, x)` and writes gradient and Hessian

# Methods
- `logpdf!(structure, out, x)`
- `grad_hess!(structure, out_grad, out_hess, x)`

All methods expect pre-allocated containers of appropriate dimensions.
"""
struct InplaceLogpdfGradHess{F,GH}
    logpdf!::F
    grad_hess!::GH
end

"""
    InplaceLogpdfGradHess(logpdf!, grad!, hess!)

Outer convenience constructor that accepts separate `grad!` and `hess!` functions.
Internally it wraps them with `NaiveGradHess` to provide a unified `grad_hess!`
implementation and returns an `InplaceLogpdfGradHess` instance.

# Arguments
- `logpdf!`: Function `(out_logpdf, x) ->` writes the log-density into `out_logpdf`
- `grad!`: Function `(out_grad, x) ->` writes the gradient into `out_grad`
- `hess!`: Function `(out_hess, x) ->` writes the Hessian into `out_hess`

# See also
- `NaiveGradHess` — adapter that combines separate `grad!`/`hess!` into `grad_hess!`.
"""
function InplaceLogpdfGradHess(logpdf!::F, grad!::G, hess!::H) where {F,G,H}
    wrapper_grad_hess! = NaiveGradHess(grad!, hess!)
    return InplaceLogpdfGradHess(logpdf!, wrapper_grad_hess!)
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
    grad_hess!(inplace::InplaceLogpdfGradHess, out_grad, out_hess, x)

Evaluate the gradient and the Hessian at point `x`, writing the result to pre-allocated containers `out_grad` and `out_hess`.
"""
function grad_hess!(inplace::InplaceLogpdfGradHess, out_grad, out_hess, x)
    inplace.grad_hess!(out_grad, out_hess, x)
    return out_grad, out_hess
end
