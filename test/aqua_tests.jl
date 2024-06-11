@testitem "Aqua: Auto QUality Assurance" begin
    using Aqua, ExponentialFamilyProjection

    Aqua.test_all(ExponentialFamilyProjection; ambiguities = false, deps_compat = (; check_extras = false, check_weakdeps = true))
end