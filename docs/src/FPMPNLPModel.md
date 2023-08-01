# **FPMPNLPModel**: Multi-Precision Model


`FPMPNLPModel` (Floating Point Multi Precision Non Linear Model) is a structure meant to "augment" the `NLPModel` ([NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)) structure and interfaces to deal with multiple floating point formats and handle evaluation errors.

# Fields

`FPMPNLPModel` structure contains a `NLPModel` field and additional field related to multiple precision.
The types are:
* `D < AbstractFloat`
* `S < AbstractVector`
* `H < AbstractFloat`
* `B < Tuple}`

| Field | Type | Notes |
|-----|----|-----|
  `Model`|`AbstractNLPModel{D, S}` | Base model
  `meta` | `NLPModelMeta` | meta data | Base model meta data
  `counters`|`MPCounters` | multi-precision counters
  `FPList`|`Vector{DataType}` | List Floating Point formats used 
  `EpsList`|`Vector{H}` | List of epsilon machine corresponding to `FPList`
  `UList`|`Vector{H}` | List of unit roundoff corresponding to `FPList`
  `OFList`|`Vector{H}` | List of largest representable numbers corresponding to `FPList`
  `γfunc`| | dot product error model function callback.
  `ωfRelErr`|`Vector{H}` | List of relative error factor for objective function evaluation corresponding to FP formats in `FPList`
  `ωgRelErr`|`Vector{H}` |List of relative error factor for gradient evaluation corresponding to FP formats in `FPList`
  `ObjEvalMode` |`Int` | objective function error evaluation mode
  `GradEvalMode` |`Int` | gradient error evaluation mode
  `X`| `B` | Container used for interval evaluation, memory pre-allocation
  `G`| `B` | Container used for interval evaluation, memory pre-allocation

## `γfunc`

 `γfunc` callback function provides the error on dot product: $|x.y - fl(x.y)|\leq |x|.|y|\gamma func(n,u)$ with $n$ the dimension of $x$ and $y$ vector and $u$ the unit-roundoff of the FP format used to perform the dot product operation.
 The expected template is: `γfunc(n::Int,u::AbstractFloat)` with `n` the dimension of the problem and `u` the unit roundoff of the considered FPFormat.
 By default, `γfunc(n,u) = n*u` is used.
This function is used to take finite-precision norm computation error into account, to further guarantee gradient error bounds (details below).

# Constructors

1. `FPMPNLPModel(Model::AbstractNLPModel{D,S},FPList::Vector{K}; kwargs...) where {D,S,K<:DataType}`

2. `FPMPNLPModel(f,x0, FPList::Vector{DataType}; kwargs...)`
Build a `ADNLPModel` from `f` and `x0` and call constructor 1.

## Keyword arguments
  + `HPFormat=Float64` : high precision format (must be at least as accurate as `FPList[end]`), corrensponds to `H` parameter after instanciation
  + `γfunc=nothing` : use default callback if not provided (see Fields section above)
  + `ωfRelErr=HPFormat.(sqrt.(eps.(FPList)))`: use relative error model by default for objective evaluation
  + `ωgRelErr=HPFormat.(sqrt.(eps.(FPList)))`: use relative error model by default for gradient evaluation
  + `obj_int_eval = false` : if true, use interval arithmetic for objective value and error evaluation
  + `grad_int_eval = false` : if true, use interval arithmetic for gradient value and error evaluation

## Checks upon instanciation
Some checks are performed upon instanctiation. These checks include:
+ Length consistency of vector fields:  `FPList`, `EpsList`, `UList`
+ `HPFormat` is at least as accurate as the highest precision floating point format in `FPList`. Ideally HPFormat is more accurate to ensure the numerical stability.
+ Interval evaluations: it might happen that interval evaluation of objective function and/or gradient is type-unstable or returns an error. The constructor returns an error in this case. This type of error is most likely due to [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package.
+ `FPList` is ordered by increasing floating point format accuracy

This checks can return `@warn` or `error`.

# Evaluation Errors

`FPMPNLPModel` provides interfaces to evaluate the objective function and gradient and evaluation errors due to finite-precision computations.
The evaluation errors on the objective function and the gradient, $\omega_f$ and $\omega_g$, are such that
* Objective function : $|f(x) - fl(f(x))|\leq \omega_f$
* Gradient : $||\nabla f(x) - fl(\nabla f(x))||_2 \leq \omega_g ||fl(\nabla f(x))||_2$
where $fl()$ denotes the finite-precision computation with one of the FP formats in `FPList`.

`FPNLPModel` enables to determine evaluation errors $\omega_f$ and $\omega_g$ either:
* based on relative error model with `ωfRelErr` and `ωgRelErr`,
* in a guaranteed way based on interval evaluation with `IntervalArithmetic.jl` package.
By default, relative error model is used. Interval evaluation can be selected upon instanciation of `FPMPNLPModel` with `obj_int_eval` and `grad_int_eval` kwargs.


## Taking Norm Computation Error Into Account
For the gradient error, the 2-norm computation error due to finite-precision computations is taken into account via `γfunc`, such that $\omega_g$ is guaranteed. The norm computation error is given by $| ||x||_2 - fl(||x||_2) | \leq \beta(n+2,u) fl(||x||_2)$, with $n$ the dimension of the problem, $u$ the unit-roundoff of the FP format used to perform the norm computation, and

$\beta(n,u) = \max(|\sqrt{\gamma func(n,u)-1}-1|,|\sqrt{\gamma func(n,u)+1}-1|)$ defined from `γfunc`.

The high precision format `H` is used to compute $\beta(n+2,u)$.

## Relative Error Evaluation
By default, evaluation errors for objective and gradient are estimated with the relative error model.

The error models are: 
* Objective: $|f(x) - fl(f(x))| \leq$`ωfRelErr[id]`$*fl(f(x))$ where `id` is the index of the FP format of $x$ in `FPList`. `objerrmp` returns the value of the classic evaluation of the objective as the value of the objective, and `ωfRelErr[id]`$*fl(f(x))$ as the evaluation error $\omega_f$, where `id` is the index of the FP format of $x$ in `FPList`
* Gradient: $||\nabla f(x) - fl(\nabla f(x))||_2 \leq$`ωgRelErr[id]`$||fl(\nabla f(x))||_2$ where where `id` is the index of the FP format of $x$ in `FPList`. `graderrmp` returns the value of the classic evaluation of the gradient as the value of the gradient, and `ωgRelErr[id]` as the value of the evaluation error.

See keyword arguments section for `ωfRelErr` and `ωgRelErr` default values. 

## Interval Error Evaluation

Error of the objective (resp. gradient) evaluation can be determined with interval arithmetic. This evaluation mode can be set upon `FPMPNLPModel` instanciation with `obj_int_eval` and `grad_int_eval`.

* Objective: evaluating the objective with interval arithmetic provides the interval $[\underline{f},\overline{f}]$ such that $\underline{f}\leq fl(f) \leq \overline{f}$. `objerrmp` returns the middle of the interval as the value of the objective, that is $(\underline{f}+\overline{f})/2$, and returns the diameter of the interval as $\omega_f$, that is, $\omega_f = (\underline{f}-\overline{f})/2$

* Gradient: evaluating the gradient with interval arithmetic provides the interval vector $G = [\underline{g}_1,\overline{g}_1] \times ... \times [\underline{g}_n,\overline{g}_n]$ such that $\nabla f(x) \in G$. `graderrmp` returns $g$ the middle of the interval vector as the value of the gradient. The evaluation error $\nabla f(x)-g$ is the vector of the diameters of the element of $G$. The error $\omega_g$ returned by `graderrmp` is $||\nabla f(x)-g||_2/||g||_2 * (1+\beta(n+2,u))/(1-\beta(n+2,u))$, the second term takes norm computation error into account.

**Warning**
* Interval evaluation is slow compared with "classic" evaluation.
* Although guaranteed, interval bounds can be quite pessimistic.
* Interval evaluation might fail with rounding mode other than `:accurate` for FP formats other than `Float32` and `Float64`. When using interval evaluation, it is recommended to call 
```julia
using IntervalArithmetic
setrounding(Interval,:accurate)
``` 
before instanciating a `FPMPNLPModel`.

## HPFormat
`FPMPNLPModel` requires a high-precision FP format, given by `HPFormat` constructor's keyword argument. This format is used to compute accurately a bound on finite-precision norm evaluation error, to further guaranteed the bound $\omega_g$ in interval evaluation context.
The bound on norm error is computed via `γfunc`.

# Interface

|function|desctiption|
|--------|-------|
`FPMPNLPModel`| constructor (see Constructor section)
`get_id`| Returns index of FP format in `FPList`
`objerrmp`| Compute objective function value and evaluation error $\omega_f$
`graderrmp!` | Compute gradient and evaluation error $\omega_g$ (no memory allocation for gradient)
`graderrmp` | Compute gradient and evaluation error $\omega_g$ (memory allocation for gradient)
`objReachPrec` | Compute objective and evaluation error $\omega_f$, increases FP format precision until bound on $\omega_f$ is reached
`gradReachPrec!` | Compute gradient and evaluation error $\omega_g$, increases FP format precision until bound on $\omega_g$ is reached (no mem. allocation for gradient)
`gradReachPrec` | Compute gradient and evaluation error $\omega_g$, increases FP format precision until bound on $\omega_g$ is reached (mem. allocation for gradient)

# Examples


## Interval Evaluations
```@example
using MultiPrecisionR2
using IntervalArithmetic

setrounding(Interval,:accurate)
Formats = [Float16,Float32,Float64] # FP formats
f(x) = sum(x.^2) # objective function
dim = 100 # problem dimentsion
x0 = ones(100)
mpmodel = FPMPNLPModel(f,x0,Formats) # create multi-precision model, will use interval arithmetic for evaluation error.

x16 = ones(Float16,dim)
x32 = Float32.(x16)
x64 = x0

# objective evaluation 
f16, ωf16 = objerrmp(mpmodel,x16) # Float16 objective evaluation
f32, ωf32 = objerrmp(mpmodel,x32) # Float32 objective evaluation
f64, ωf64 = objerrmp(mpmodel,x64) # Float32 objective evaluation

x = (x16,x32,x64) # element of the tuple should refer to the same vector in different FP formats
bound = ωf32*1.1 # error bound reachable with Float32 precision
fx, ωfx, fid = objReachPrec(mpmodel,x,bound; π=1) # evaluate objective with increasing precision, starting with mpnlpmodel.FPFormat[π] = Float16, until evaluation error is lower than bound (satisfied with Float32)


# gradient evaluation
g16, ωg16 = graderrmp(mpmodel,x16) # Float16 gradient evaluation
g32, ωg32 = graderrmp(mpmodel,x32) # Float32 gradient evaluation
g64, ωg64 = graderrmp(mpmodel,x64) # Float32 gradient evaluation

x = (x16,x32,x64) # element of the tuple should refer to the same vector in different FP formats
bound = ωg64*1.1 # error bound reachable with Float64 precision
gx, ωgx, gid = gradReachPrec(mpmodel,x,bound; π=1) # evaluate gradient with increasing precision, starting with mpnlpmodel.FPFormat[π] = Float16, until evaluation error is lower than bound (satisfied with Float64)
```

## Relative error

```@example
using MultiPrecisionR2

Formats = [Float16,Float32,Float64] # FP formats
f(x) = sum(x.^2) # objective function
dim = 100 # problem dimentsion
x0 = ones(100)
ω = Float64.([sqrt(eps(t)) for t in Formats]) # relative errors, have to be H format (Float64)
mpmodel = FPMPNLPModel(f,x0,Formats; ωfRelErr = ω, ωgRelErr = ω) # create multi-precision model, will use relative error model base on ωfRelErr and ωgRelErr.

x16 = ones(Float16,dim)
x32 = Float32.(x16)
x64 = x0

# objective evaluation 
f16, ωf16 = objerrmp(mpmodel,x16) # Float16 objective evaluation
f32, ωf32 = objerrmp(mpmodel,x32) # Float32 objective evaluation
f64, ωf64 = objerrmp(mpmodel,x64) # Float64 objective evaluation

# gradient evaluation
g16, ωg16 = graderrmp(mpmodel,x16) # Float16 gradient evaluation
g32, ωg32 = graderrmp(mpmodel,x32) # Float32 gradient evaluation
g64, ωg64 = graderrmp(mpmodel,x64) # Float64 gradient evaluation
```