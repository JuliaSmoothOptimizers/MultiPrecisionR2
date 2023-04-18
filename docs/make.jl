using MultiPrecisionR2
using Documenter

DocMeta.setdocmeta!(JSOTemplate, :DocTestSetup, :(using MultiPrecisionR2); recursive = true)

makedocs(;
  modules = [MultiPrecisionR2],
  doctest = true,
  linkcheck = false,
  strict = false,
  authors = "Dominique Monnet <monnetdo@gmail.com> and contributors",
  repo = "",
  sitename = "MultiPrecisionR2.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "",
    assets = ["assets/style.css"],
  ),
  pages = ["Home" => "index.md", "Reference" => "reference.md"],
)

deploydocs(;
  repo = "",
  push_preview = true,
  devbranch = "main",
)
