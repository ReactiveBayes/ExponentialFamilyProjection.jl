module ExponentialFamilyProjection

using ExponentialFamily,
    ExponentialFamilyManifolds,
    BayesBase,
    Distributions,
    ManifoldsBase,
    Manifolds,
    Static,
    StatsFuns,
    LinearAlgebra,
    FastCholesky,
    Bumper,
    StaticArrays,
    Random

import BayesBase: InplaceLogpdf

# This can go to the ExponentialFamily.jl, very useful
# The idea here is that it is not necessary to pack a tuple of numbers into a vector
__projection_fast_pack_parameters(t::NTuple{N,<:Number}) where {N} = t
__projection_fast_pack_parameters(t) = ExponentialFamily.pack_parameters(t)

include("manopt/bounded_norm_update_rule.jl")
include("cvi.jl")
include("strategies/control_variate.jl")
include("strategies/mle.jl")
include("projected_to.jl")

end
