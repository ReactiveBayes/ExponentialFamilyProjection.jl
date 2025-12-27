module DifferentiationInterfaceExt

using ExponentialFamilyProjection
using DifferentiationInterface
using ADTypes
using BayesBase

# Import the types we need
import ExponentialFamilyProjection: InplaceLogpdfGradHess, NaiveGradHess
import BayesBase: ContinuousUnivariateLogPdf, ContinuousMultivariateLogPdf

"""
    ExponentialFamilyProjection.to_inplace_gradhess(
        argument::ContinuousUnivariateLogPdf,
        backend::ADTypes.AbstractADType = AutoForwardDiff()
    ) -> InplaceLogpdfGradHess

Converts a `ContinuousUnivariateLogPdf` to an `InplaceLogpdfGradHess` object using automatic
differentiation via DifferentiationInterface.jl.

This function creates optimized gradient and Hessian computations using the specified AD backend.
The resulting `InplaceLogpdfGradHess` can be used with gradient-based projection strategies
like `BonnetStrategy` or `GaussNewton`.

# Arguments
- `argument::ContinuousUnivariateLogPdf`: The continuous univariate log-pdf to convert
- `backend::ADTypes.AbstractADType`: The AD backend to use (default: `AutoForwardDiff()`)

# Example
```julia
using ExponentialFamily, BayesBase
using DifferentiationInterface
using ADTypes

# Define a logpdf function
my_logpdf(x) = logpdf(Normal(0, 1), x)
my_continuous_logpdf = ContinuousUnivariateLogPdf(my_logpdf)

# Convert with default ForwardDiff backend
inplace = to_inplace_gradhess(my_continuous_logpdf)

# Or use a different backend
using Zygote
inplace = to_inplace_gradhess(my_continuous_logpdf, AutoZygote())

# Use in projection
params = ProjectionParameters(strategy = GaussNewton(nsamples = 1))
prj = ProjectedTo(NormalMeanVariance; parameters = params)
result = project_to(prj, inplace)
```
"""
function ExponentialFamilyProjection.to_inplace_gradhess(
    argument::ContinuousUnivariateLogPdf,
    backend::ADTypes.AbstractADType = AutoForwardDiff(),
)
    # Extract the base logpdf function
    base_logpdf = argument.logpdf

    # Create inplace logpdf wrapper
    function logpdf_inplace!(out, x)
        out[1] = base_logpdf(x)
        return out
    end

    # For univariate case, we need derivative (not gradient)
    # and second derivative (not Hessian)
    function grad_inplace!(out, x)
        # derivative returns a scalar, we need to put it in out
        out[1] = derivative(base_logpdf, backend, x)
        return out
    end

    function hess_inplace!(out, x)
        # Second derivative - also a scalar
        out[1] = derivative(y -> derivative(base_logpdf, backend, y), backend, x)
        return out
    end

    # Combine gradient and Hessian into a unified grad_hess! function
    grad_hess_wrapper! = NaiveGradHess(grad_inplace!, hess_inplace!)

    return InplaceLogpdfGradHess(logpdf_inplace!, grad_hess_wrapper!)
end

"""
    ExponentialFamilyProjection.to_inplace_gradhess(
        argument::ContinuousMultivariateLogPdf,
        backend::ADTypes.AbstractADType = AutoForwardDiff()
    ) -> InplaceLogpdfGradHess

Converts a `ContinuousMultivariateLogPdf` to an `InplaceLogpdfGradHess` object using automatic
differentiation via DifferentiationInterface.jl.

This function creates optimized gradient and Hessian computations using the specified AD backend,
with preparation/caching for better performance on repeated evaluations.

# Arguments
- `argument::ContinuousMultivariateLogPdf`: The continuous multivariate log-pdf to convert
- `backend::ADTypes.AbstractADType`: The AD backend to use (default: `AutoForwardDiff()`)

# Example
```julia
using ExponentialFamily, BayesBase
using DifferentiationInterface
using ADTypes
using DomainSets

# Define a multivariate logpdf function
my_logpdf(x) = logpdf(MvNormal([0.0, 0.0], [1.0 0.0; 0.0 1.0]), x)
my_continuous_logpdf = ContinuousMultivariateLogPdf(ℝ^2, my_logpdf)

# Convert with default ForwardDiff backend
inplace = to_inplace_gradhess(my_continuous_logpdf)

# Use in projection
params = ProjectionParameters(strategy = GaussNewton(nsamples = 1))
prj = ProjectedTo(MvNormalMeanCovariance, 2; parameters = params)
result = project_to(prj, inplace)
```
"""
function ExponentialFamilyProjection.to_inplace_gradhess(
    argument::ContinuousMultivariateLogPdf,
    backend::ADTypes.AbstractADType = AutoForwardDiff(),
)
    # Extract the base logpdf function
    base_logpdf = argument.logpdf

    # Create inplace logpdf wrapper
    function logpdf_inplace!(out, x)
        out[1] = base_logpdf(x)
        return out
    end

    # Create inplace gradient wrapper using DifferentiationInterface
    function grad_inplace!(out, x)
        gradient!(base_logpdf, out, backend, x)
        return out
    end

    # Create inplace Hessian wrapper using DifferentiationInterface
    function hess_inplace!(out, x)
        hessian!(base_logpdf, out, backend, x)
        return out
    end

    # Combine gradient and Hessian into a unified grad_hess! function
    grad_hess_wrapper! = NaiveGradHess(grad_inplace!, hess_inplace!)

    return InplaceLogpdfGradHess(logpdf_inplace!, grad_hess_wrapper!)
end

end # module
