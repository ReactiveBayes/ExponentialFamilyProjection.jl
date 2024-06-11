module ExponentialFamilyProjection

using ExponentialFamily,
    ExponentialFamilyManifolds,
    BayesBase,
    Distributions,
    ManifoldsBase,
    Manifolds,
    Static,
    StaticArrays,
    Random

import BayesBase: InplaceLogpdf

include("cvi.jl")
include("projected_to.jl")

end
