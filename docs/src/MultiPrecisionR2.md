# MutiPrecisionR2

`MutliPrecisionR2.jl` implements MPR2 (Multi-Precision Quadratic Regularization) algorithm, a multiple Floating Point (FP) precision (or formats) adaptation of the Quadratic Regularization (R2) algorithm. R2 is a first order algorithm designed to solve unconstrained minimization problems

$\min_x f(x)$,

with $f:\mathbb{R}^n \rightarrow \mathbb{R}$ a smooth non-linear function and $x \in \mathbb{R}^n$.

MPR2 extends R2 by dynamically adapting the FP formats used for objective and gradient evaluations so that the convergence is guaranteed in spite of evaluation errors due to finite-precision computations and over/underflow are avoided.

MPR2 relies on `FPMPNLPModel` structure (see documentation) to evaluate the objective and gradient with multiple floating point format and control the evaluation errors.

# How to Run

MPR2 algorithm is run in the same fashion than solvers from [`JSOSolvers.jl`](https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl) package, but uses a `FPMPNLPModel` structure which is a multi-precision extension of `NLPModel` structure to multi-precision.

MPR2 algorithm is run with `MPR2()` function, which returns a [`GenericExecutionStat`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl) structure containing useful information (nb. of iteration, termination status, etc.). The number of evaluations can be access via the `FPMPNLPModel` structure (see `FPMPNLPModel` and `MPCounters` documentation).
```@example
using MultiPrecisionR2

T = [Float16,Float32] # floating point formats used for evaluation
f(x) = sum(x.^2) # objective function
n = 100 # problem dimension
x0 = ones(Float32,n) # initial solution
mpnlp = FPMPNLPModel(f,x0,T) # creates a multi-precision model of the problem
stats = MPR2(mpnlp)
println("Objective was evaluated $(neval_obj(mpnlp,Float32)) times with Float32 and $(neval_obj(mpnlp,Float64)) times with Float64")
println("Gradient was evaluated $(neval_grad(mpnlp,Float32)) times with Float32 and $(neval_grad(mpnlp,Float64)) times with Float64")
```

# Solver Options

Some parameters of the algorithm can be given as keyword arguments of `MPR2()`. The type parameters are `S::AbstractVector`, `H::AbstractFloat`, `T::DataType`, `E::DataType`.

`MPR2(mpnlp::FPMPNLPModel; kwargs...)`

|kwarg|description|
|-----|-----------|
`x₀::S = MPnlp.Model.meta.x0` | initial guess, FP format must be in `mpnlp.FPList` 
`par::MPR2Params = MPR2Params(MPnlp.FPList[1],H)` | MPR2 parameters, see `MPR2Params` for details
`atol::H = H(sqrt(eps(T)))` | absolute tolerance on first order criterion 
`rtol::H = H(sqrt(eps(T)))` | relative tolerance on first order criterion
`max_eval::Int = -1` | maximum number of evaluation of the objective function.
`max_iter::Int = 1000` | maximum number of iteration allowed
`σmin::T = sqrt(T(MPnlp.EpsList[end]))` | minimal value for regularization parameter. Value must be representable in any of the floating point formats of MPnlp. 
`verbose::Int=0` | display iteration information if > 0
`e::E` | user defined structure, used as argument for `compute_f_at_x!`, `compute_f_at_c!`, `compute_g!` and `recompute_g!` callback functions.
`compute_f_at_x!` | callback function to select precision and compute objective value and error bound at the current point. Allows to reevaluate the objective at x if more precision is needed.
`compute_f_at_c!` | callback function to select precision and compute objective value and error bound at candidate.
`compute_g!` | callback function to select precision and compute gradient value and error bound. Called at the end of main loop.
`recompute_g!` | callback function to select precision and recompute gradient value if more precision is needed. Called after step, candidate and model decrease computation in main loop.
`selectPic!` | callback function to select FP format of `c` at the next iteration

# Choosing MPR2 Parameters

MPR2 algorithm parameters for error tolerance and step acceptance can be given as `par::MPR2Params` keyword argument.
These parameters correspond to `MPR2Params` fields, with type parameters `H::AbstractFloat` and `L::AbstractFloat`.

|Field|Default value|Description|
|-----|-----------|-------------|
`η₀::H` | `0.01` |  controls objective function error tolerance, convergence condition is ωf ≤ η₀ ΔT (see `FPMPNLPModel` for details on ωf) 
`η₁::H` | `0.02` | step successful if ρ ≥ η₁ (update incumbent)
`η₂::H` | `0.95` | step very successful if ρ ≥ η₂ (decrease σ ⟹ increase step length)
`κₘ::H` | `0.02` | tolerance on gradient evaluation error, μ ≤ κₘ (see `computeMu`) 
`γ₁::L` | `2^(-2)` | σk+1 = σk * γ₁ if ρ ≥ η₂
`γ₂::L` | `2` | σk+1 = σk * γ₂ if ρ < η₁

These parameters must satisfy some conditions, see `MPR2Params` for details. These conditions can be checked with `CheckMPR2ParamConditions()` function.

# Evaluation Error Mode

MPR2 convergence is ensured by taking into account objective and gradient evaluation errors. These errors can be evaluated with interval arithmetic of based on relative error assumption. The error evaluation mode is chosen upon the `FPMPNLPModel` instanciation, given as argument of `MPR2()`. Evaluating the objective/gradient and the error is done with the interfaces provided in `MPNLPModels.jl`, see `FPMPNLPModel` documentation. 

# Callbacks for Personalized Implementation

`MPR2()` allows the user to define its own strategies for evaluation FP formats selection and error handling for objective and gradient evaluations. This can be done by providing the callbacks `compute_f_at_x!`, `compute_f_at_c!`, `compute_g!`, `recompute_g!` and `selectPic!`.
See "MPR2 advanced use" tutorial for usage. 
