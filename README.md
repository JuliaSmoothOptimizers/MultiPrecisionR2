# MultiPrecisionR2 #

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaSmoothOptimizers.github.io/MultiPrecisionR2/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaSmoothOptimizers.github.io/MultiPrecisionR2/dev)
[![Build Status](https://github.com/JuliaSmoothOptimizers/MultiPrecisionR2/workflows/CI/badge.svg)](https://github.com/JuliaSmoothOptimizers/MultiPrecisionR2/actions)
[![Build Status](https://api.cirrus-ci.com/github/JuliaSmoothOptimizers/MultiPrecisionR2.svg)](https://cirrus-ci.com/github/JuliaSmoothOptimizers/MultiPrecisionR2)
[![Coverage](https://codecov.io/gh/JuliaSmoothOptimizers/MultiPrecisionR2/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaSmoothOptimizers/MultiPrecisionR2)

`MultiPrecisionR2.jl` is a package that implements a multi-precision version of the Quadratic Regularization (R2) algorithm, a first-order algorithm, for solving non-convex, continuous, smooth optimization problems. The Floating Point (FP) format is adapted dynamically during algorithm execution to use low precision FP formats as much as possible while ensuring convergence and numerical stability.

The package also implements multi-precision models `FPMPNLPModel` structure that is meant to extend the `NLPModel`(see [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)) to multiple precision. The interfaces for objective and gradient evaluations are extended to provide evaluation errors.

`MultiPrecisionR2` can ensure numerical stability by using interval evaluations of the objective function and gradient. Interval evaluation relies on [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package to perform the interval evaluations.

## Installation
```julia
using Pkg
Pkg.add("MultiPrecisionR2")
```
## Minimal examples
```julia
using MultiPrecisionR2
using ADNLPModels
using IntervalArithmetic

setrounding(Interval,:accurate) # set interval mode for guaranteed evaluation of the objective function and the gradient 
FP = [Float16,Float32] # define floating point formats used by the algorithm for objective and gradient evaluation
f(x) = x[1]^2 + x[2]^2 # some objective function
x0 = ones(Float32,2) # initial point
mpmodel = FPMPNLPModel(f,x0,FP); # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)
stats = MPR2(mpmodel) # run the algorithm
```

```julia
using MultiPrecisionR2
using ADNLPModels
using IntervalArithmetic
using OptimizationProblems
using OptimizationProblems.ADNLPProblems

setrounding(Interval,:accurate)
FP = [Float16,Float32] # define floating point formats used by the algorithm for objective and gradient evaluation
s = :woods # select problem
nlp = eval(s)(n=12,type = Val(FP[end]), gradient_backend = ADNLPModels.GenericForwardDiffADGradient)
mpmodel = FPMPNLPModel(nlp,FP); # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)
stats = MPR2(mpmodel) # run the algorithm
```

## Warnings
1. MultiPrecisionR2 is designed to work with FP formats. Other format, such as fix point, might break the convergence property of the algorithm.
2. Unfortunately, other modes than `:accurate` for interval evaluation are not guaranteed to work with FP formats different from `Float32` and `Float64`. 
3. If interval evaluation mode is used, interval evaluations of the objective and the gradient are automatically tested upon `FPMPNLPModel` instanciation.  An error is thrown if the evaluation fails. This might happen for several reasons related to [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package.


## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/MultiPrecisionR2/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.
