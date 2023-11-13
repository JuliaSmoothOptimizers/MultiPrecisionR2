using MultiPrecisionR2
using Documenter

DocMeta.setdocmeta!(MultiPrecisionR2, :DocTestSetup, :(using MultiPrecisionR2); recursive = true)

makedocs(;
  modules = [MultiPrecisionR2],
  doctest = true,
  linkcheck = false,
  authors = "Dominique Monnet <monnetdo@gmail.com> and contributors",
  repo = "https://github.com/JuliaSmoothOptimizers/MultiPrecisionR2",
  sitename = "MultiPrecisionR2.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "",
    assets = ["assets/style.css"],
  ),
  pages = [
    "Home" => "index.md",
    "Reference" => "reference.md",
    "MPCounters" => "MPCounters.md",
    "FPMPNLPModel" => "FPMPNLPModel.md",
    "MultiPrecisionR2" => "MultiPrecisionR2.md",
    "FPMPNLPModel Tutorial" => "tutorial_FPMPNLPModel.md",
    "MPR2 Tutorial: Basic Use " => "tutorial_MPR2_basic_use.md",
    "MPR2 Tutorial: Advanced Use " => "tutorial_MPR2_advanced_use.md",
  ],
)

deploydocs(;
  repo = "github.com/JuliaSmoothOptimizers/MultiPrecisionR2.git",
  push_preview = true,
  devbranch = "main",
)
