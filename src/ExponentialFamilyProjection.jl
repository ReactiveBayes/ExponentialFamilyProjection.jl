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

include("strategies/control_variate.jl")

include("cvi.jl")
include("projected_to.jl")

end
