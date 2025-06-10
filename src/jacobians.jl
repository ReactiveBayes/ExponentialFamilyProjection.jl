function jacobian_nat_to_manifold!(::AbstractManifold, X_p, X_nat)
    X_p .= X_nat
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.NormalMeanVariance}, X_p, X_nat) where {F}
    X_p[1:1] .= X_nat[1]
    X_p[2:2] .= -X_nat[2]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.Gamma}, X_p, X_nat) where {F}
    X_p[1:1] .= X_nat[1]
    X_p[2:2] .= -X_nat[2]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.Rayleigh}, X_p, X_nat) where {F}
    X_p[1:1] .= -X_nat[1]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.Geometric}, X_p, X_nat) where {F}
    X_p[1:1] .= -X_nat[1]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.GammaInverse}, X_p, X_nat) where {F}
    X_p[1:1] .= -X_nat[1]
    X_p[2:2] .= -X_nat[2]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.Exponential}, X_p, X_nat) where {F}
    X_p[1:1] .= -X_nat[1]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.Weibull}, X_p, X_nat) where {F}
    X_p[1:1] .= -X_nat[1]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.Laplace}, X_p, X_nat) where {F}
    X_p[1:1] .= -X_nat[1]
    return X_p
end

function jacobian_nat_to_manifold!(::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.LogNormal}, X_p, X_nat) where {F}
    X_p[1:1] .= X_nat[1]
    X_p[2:2] .= -X_nat[2]
    return X_p
end

function jacobian_nat_to_manifold!(M::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.MvNormalMeanScalePrecision}, X_p, X_nat) where {F}
    k = first(ExponentialFamilyManifolds.getdims(M))
    X_p[1:k] .= X_nat[1:k]
    X_p[k+1:k+1] .= -X_nat[k+1:k+1]
    return X_p
end

function jacobian_nat_to_manifold!(M::ExponentialFamilyManifolds.NaturalParametersManifold{F, ExponentialFamily.MvNormalMeanCovariance}, X_p, X_nat) where {F}
    k = first(ExponentialFamilyManifolds.getdims(M))
    X_p[1:k] .= X_nat[1:k]
    X_p[(k + 1):(k + k^2)] .= -X_nat[(k + 1):(k + k^2)]
    return X_p
end