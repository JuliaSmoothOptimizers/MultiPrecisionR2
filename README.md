# MultiPrecisionR2.jl #

MultiPrecisionR2 is a package that implements a multi-precion version of the Quadratic Regularization (R2) algorithm for solving non-convex, continuous, smooth optimization problems. The evaluation precision of the objective function and the gradient is adapted dynamically to use low precision Floating Point (FP) as much as possible while ensuring convergence and numerical stability.

MultiPrecisionR2 relies on [Julia Smooth Optimizers (JSO)](https://github.com/JuliaSmoothOptimizers) environment, and in particular works with `NLPModel` implemented in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) package.

MultiPrecisionR2 ensures numerical stability by using interval evaluations of the objective function and gradient. Interval evaluation relies on [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package to perform the interval evaluation.

## Installation
```julia
using Pkg
Pkg.add("MultiPrecisionR2.jl")
```
## Minimal examples
```julia
using MultiPrecisionR2
using ADNLPModels
using IntervalArithmetic

setrounding(Interval,:accurate)
FP = [Float16,Float32] # define floating point formats used by the algorithm for objective and gradient evaluation
f4(x) = x[1]^2 + x[2]^2 # some objective function
x₀ = ones(2) # initial point
nlpList = [ADNLPModel(f4,fp.(x₀)) for fp in FP] # instanciate a list of ADNLPModel, one for each floating point format
mpmodel = FPMPNLPModel(nlpList) # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)
solver = MPR2Solver(mpmodel); # instaciate the algorithm structure
stat = solve!(solver,mpmodel) # run the algorithm
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
nlpList = [eval(s)(n=12,type = Val(F)) for F ∈ FP] # instanciate a list of ADNLPModel, one for each floating point format
mpmodel = FPMPNLPModel(nlpList) # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)
solver = MPR2Solver(mpmodel); # instaciate the algorithm structure
stat = solve!(solver,mpmodel) # run the algorithm
```

## Warnings
1. MultiPrecisionR2.jl works only with Floating Point formats.
2. Unfortunately, other modes than `:accurate` for interval evaluation are not guaranteed to work with FP formats different from Float32 and Float64. 
3. If interval evaluation mode is used, interval evaluation for the objective and the gradient is automatically tested upon FPMPNLPModel instanciation.  An error is thrown if the evaluation fails. This might happen for several reasons related to [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package.


## Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/JSOTemplate.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers) organization, so questions about any of our packages are welcome.
