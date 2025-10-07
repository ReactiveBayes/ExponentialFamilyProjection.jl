"""
    NaiveGradHess(grad!, hess!)

Adapter that exposes only `grad_hess!` by calling provided `grad!` and `hess!` sequentially.
Useful as a fallback when a combined implementation is not available.
"""
struct NaiveGradHess{G,H}
    grad!::G
    hess!::H
end

"""
    grad_hess!(inplace::NaiveGradHess, out_grad, out_hess, x)

Evaluate the gradient and the Hessian at point `x` using the provided separate implementations.
"""
function grad_hess!(inplace::NaiveGradHess, out_grad, out_hess, x)
    inplace.grad!(out_grad, x)
    inplace.hess!(out_hess, x)
    return out_grad, out_hess
end

function (inplace::NaiveGradHess)(out_grad, out_hess, x)
    inplace.grad!(out_grad, x)
    inplace.hess!(out_hess, x)
    return out_grad, out_hess
end