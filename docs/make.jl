push!(LOAD_PATH, "../src/")

using Documenter, TensorAlgebra

makedocs(sitename="TensorAlgebra.jl",
    clean=true,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        footer=nothing
    )
)