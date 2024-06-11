using ExponentialFamilyProjection

DocMeta.setdocmeta!(
    ExponentialFamilyProjection,
    :DocTestSetup,
    :(using ExponentialFamilyProjection);
    recursive = true,
)

makedocs(;
    modules = [ExponentialFamilyProjection],
    authors = "Lazy Dynamics <info@lazydynamics.com>",
    sitename = "ExponentialFamilyProjection.jl",
    format = Documenter.HTML(;
        canonical = "https://lazydynamics.github.io/ExponentialFamilyProjection.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(;
    repo = "github.com/lazydynamics/ExponentialFamilyProjection.jl",
    devbranch = "main",
)
