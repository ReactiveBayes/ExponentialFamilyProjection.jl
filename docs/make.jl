using Documenter, ExponentialFamilyProjection

DocMeta.setdocmeta!(
    ExponentialFamilyProjection,
    :DocTestSetup,
    :(using ExponentialFamilyProjection);
    recursive = true,
)

makedocs(;
    modules = [ExponentialFamilyProjection],
    authors = "Mykola Lukashchuk <m.lukashchuk@tue.nl>, Dmitry Bagaev <bvdmitri@gmail.com>, Albert Podusenko <albert@lazydynamics.com>",
    warnonly = false,
    sitename = "ExponentialFamilyProjection.jl",
    format = Documenter.HTML(;
        canonical = "https://reactivebayes.github.io/ExponentialFamilyProjection.jl",
        edit_link = "main",
        assets = String[],
        repolink = "https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl",
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(;
    repo = "github.com/ReactiveBayes/ExponentialFamilyProjection.jl",
    devbranch = "main",
)
