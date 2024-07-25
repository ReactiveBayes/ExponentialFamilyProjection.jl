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

include("manopt/bounded_norm_update_rule.jl")
include("cvi.jl")
include("strategies/control_variate.jl")
include("strategies/mle.jl")
include("projected_to.jl")

end
