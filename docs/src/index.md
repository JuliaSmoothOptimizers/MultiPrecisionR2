# MultiPrecisionR2 #

`MultiPrecisionR2.jl` is a package that implements a multi-precision version of the Quadratic Regularization (R2) algorithm, a first-order algorithm, for solving non-convex, continuous, smooth optimization problems. The Floating Point (FP) format is adapted dynamically during algorithm execution to use low precision FP formats as much as possible while ensuring convergence and numerical stability.

The package also implements multi-precision models `FPMPNLPModel` structure that derives from `NLPModel` implemented in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl). The interfaces for objective and gradient evaluations are extended to provide evaluation errors.

`MultiPrecisionR2` can ensure numerical stability by using interval evaluations of the objective function and gradient. Interval evaluation relies on [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package to perform the interval evaluations.

## Installation
```julia
using Pkg
Pkg.add("MultiPrecisionR2")
```

## Minimal examples
```@example
using MultiPrecisionR2

FP = [Float16,Float32] # define floating point formats used by the algorithm for objective and gradient evaluation
f(x) = x[1]^2 + x[2]^2 # some objective function
x0 = ones(Float32,2) # initial point
mpmodel = FPMPNLPModel(f,x0,FP); # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)
stat = MPR2(mpmodel) # run the algorithm
```

```@example
using MultiPrecisionR2
using ADNLPModels
using OptimizationProblems
using OptimizationProblems.ADNLPProblems

FP = [Float16,Float32] # define floating point formats used by the algorithm for objective and gradient evaluation
s = :woods # select problem
nlp = eval(s)(n=12,type = Val(FP[end]), backend = :generic)
mpmodel = FPMPNLPModel(nlp,FP); # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)
stat = MPR2(mpmodel) # run the algorithm
```

## Warnings
1. MultiPrecisionR2 is designed to work with FP formats. Other format, such as fix point, might break the convergence property of the algorithm.
2. * Interval evaluation might fail with rounding mode other than `:accurate` for FP formats other than `Float32` and `Float64`. When using interval evaluation, it is recommended to call 
```julia
using IntervalArithmetic
setrounding(Interval,:accurate)
``` 
before instanciating a `FPMPNLPModel`.

3. If interval evaluation mode is used, interval evaluations of the objective and the gradient are automatically tested upon `FPMPNLPModel` instantiation.  An error is thrown if the evaluation fails. This might happen for several reasons related to [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package.


## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/MultiPrecisionR2/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.