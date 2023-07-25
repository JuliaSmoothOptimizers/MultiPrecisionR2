# Disclaimer

The reader is encouraged to read the `FPMPNLPModel` tutorial before this one.

# Advanced Use

`MultiPrecisionR2.jl` does more than implementing MPR2 algorithm as describes in the Basic Use tutorial. `MPR2Precision.jl` enables the user to define its own strategy to select evaluation precision levels and to handle evaluation errors. This is made possible by using callback functions when calling `MultiPrecisionR2.solve!()`. The default implementation of `MultiPrecisionR2.solve!()` relies on specific implementation of these callback functions, which are included in the package. The user is free to provide its own callback functions to change the behavior of the algorithm. 

## Diving into MPR2 Implementation
`MPR2Precision.jl` relies on callback functions that handle the objective and gradient evaluation. These callback functions are expected to compute values for the objective and the gradient and handle the evaluation precision. In the default implementation, the conditions on the error bound of objective and gradient evaluation are dealt with in these callback functions. That is why such convergence conditions (which user might choose not to implement) does not appear in the minimal implementation description of the code below.

### **Minimal Implementation Description**

Here is a minimal description of the implementation of `MultiPrecisionR2.solve!()` to understand when and why these functions are called. Please refer to the implementation for details. Note that the callback functions templates are not respected for the sake of explanation.
```julia
function solve!(solver::MPR2solver{T},MPnlp::FPMPNLPModel{H};kwargs...)
  # ...
  # some initialization (parameters, containers,...)
  # ...
  compute_f_at_x!() # initialize objective value at x0
  # check for possible overflow 
  compute_g!() # initialize gradient value at x0
  # check for possible overflow
  # ...
  # some more initialization (gradient norm tolerance,...)
  # ...
  while # main loop
    # compute step as sk = -gk/σk
    # check for step overflow
    # compute candidate as c = x+s
    # check for candidate overflow
    # compute model decrease ΔT = -g^T s
    # check for model decrease under/overflow
    recompute_g!() # possibly recompute g(x) and ωg(x) if gradient error/mu indicator is too big
    if grad_prec_fail # not enough precision with max precision
      break
    end
    if grad_recomputed# gradient recomputed by recompute_g!() 
      # recompute step with new value of the gradient
      # check overflow
      # recompute candidate with new step
      # check overflow
      # recompute model decrease with new gradient and step
    end
    compute_f_at_x() # possibly recompute f(x) and ωf(x) if ωf(x) is too big
    if f_x_prec_fail# not enough precision with max precision (ωf(x) too big)
      break
    end
    compute_f_at_c!() # compute f(c) and ωf(c)
    if f_c_prec_fail # not enough precision with max precision (ωf(c) too big)
      break
    end
    # ...
    # compute ρ = (f(x) - f(c))/ΔT
    # update x and σ
    # ... 
    if ρ ≥ η₁
      compute_g!() # compute g(c) and  ωg(c)
      # check overflow
      # ...
      # update values for next iteration
      # ...
      selectPic!()
    end
    # ...
    # check for termination
    # ...
  end # while main loop
  return status()
end
```

 The callback functions, and what they are supposed to do, are 
* `compute_f_at_x!()`: **Selects evaluation precision** and **computes objective function at the current point $x_k$** such that possible conditions on error bounds are ensured. In the main loop, although the objective at $x_k$ has already been computed at a previous iteration ($f(c_{j}) = f(x_k)$ with $j$ the last successful iteration) it might be necessary to recompute it to achieve smaller evaluation error ($\omega_f(x_k)$) if necessary. This function is also **called for initialization** (before the main loop), where typically no bound on evaluation error is required.
* `compute_f_at_c!()`: **Selects evaluation precision** and **computes objective function at the candidate $c_k$** such that possible conditions on error bounds are ensured.
* `compute_g!()`: **Selects evaluation precision** and **computes gradient at the candidate $c_k$**. It is possible here to include condiditions on gradient bound error $\omega_g(x_k)$. In the default implementation, multiple sources of rounding error are taken into account in the $\mu$ indicator that requires to compute the step, candidate and model decrease (see [Rounding Errors Handling](#rounding-errors-handling)). That is why in the default implementation no conditions on gradient error are required in this callback function, but are implemented in `recompute_g!()`.
* `recompute_g!()`: **Selects evaluation precision** and **recomputes gradient at the current point $x_k$**. This callback enables to recompute the gradient **if necessary** with a better precision if needed. In the default implementation, this callback implement the condition on the $\mu$ indicator (see [Rouding Error Handling](#rounding-errors-handling)) and implements strategies to achieve sufficiently low value of $\mu$ to ensure convergence.
* `selectPic!()`: Select FP format for $c_{k+1}$, enables to lower the FP format used for evaluation at the next iteration. See [Candidate Precision Selection](#candidate-precision-selection) section for details.

### **Callback Functions: Templates**

* `prec_fail::Bool = compute_f_at_c!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T) where {H, L, E, T <: Tuple}`
* `prec_fail::Bool = compute_f_at_x_default!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, x::T) where {H, L, E, T <: Tuple}`
* `prec_fail::Bool = compute_g!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T, g::T) where {H, L, E, T <: Tuple}`
* `prec_fail::Bool, recompute_g::Bool = recompute_g!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, x::T, g::T, s::T) where {H, L, E, T <: Tuple}`

**Arguments**:
* `m:FPMPNLPModel{H}`: multi-precision model, needed to perform objective/gradient evaluation
* `st::MPR2State{H}`: States or intermediate variables of MPR2 algorithm ($\rho$, $\mu$, ...), see `MPR2State` documentation
* `π::MPR2Precisions`: Current precision indices, see `MPR2Precisions` documentation
* `p::MPR2Params{H, L}`: algorithm parameters, see `MPR2Params` documentation
* `e::E`: user defined structure to store additional information if needed
* `x::T`: current point
* `s::T`: current step
* `c::T`: current candidate
* `g::T`: current gradient

**Outputs**
* `prec_fail::Bool`: `true` if the evaluation failed, typical reason is that not enough evaluation could be reached. If `prec_fail == true`, stops the algorithm and set `st.status = :exception`.
* `recompute_g::Bool`: `true` if `recompute_g!()` has recomputed the gradient. In this case, the set, candidate and model decrease are recomputed (see [Minimal Implementation Description](#minimal-implementation-description))

### **Callback Functions: Expected Behaviors**

**Modified Variables:** The callback functions are expected to update all the variables that they modify. These variables are typically MPR2 states in `st::MPR2State{H}` structure, the current precisions in `π::MPR2Precisions` structure, extra values in user defined `e::E` structure. The callback functions `compute_g!()` and `recompute_g!()` are also expected to modify the gradient `g::T`.

**Variable that should not be modified:** The callback functions **should not modify x, s, and c**.

Below is a table that recaps what variables each callback can/should update.
|Callback| Modified Variables|
| ------ | ----------------- |
|`compute_f_at_x!()`| `st.status`, `st.f`, `st.ωf`, `π.πf`
|`compute_f_at_c!()`| `st.status`, `st.f⁺`, `st.ωf⁺`, `π.πf⁺`
|`compute_g!()`| `st.status`, `g`, `st.ωg`, `π.πg`
|`recompute_g()`| `st.status`, `g`, `st.ωg`, `π.πg`, `st.ΔT`, `π.ΔT`, `st.x_norm`, `π.πnx`, `st.s_norm`, `π.πns`, `st.ϕ`, `st.ϕhat`, `π.πc`, `st.μ`

### **Multi-Precision Evaluation and Vectors Containers**
MPR2 performs objective and gradient evaluations and error bounds estimation with different FP format. These evaluations are performed with `FPMPNLPModels.objerrmp()` and `FPMPNLPModels.graderrmp!()` functions (see `FPMPNLPModels` documentation and [FPMPNLPModel section](#fpmpnlpmodel-creating-a-multi-precision-model)). These functions expect as input a `Vector{S}` where to evaluate the objective/gradient where `S` is the FP format that matches `FPMPNLPModel.FPList[id]` the FP format of index `id`. 
That is, it is not possible to perform an evaluation in an FP format different than the FP format of the input vector. This further means that, when MPR2 is running, it is necessary to change the FP format of $x_k$, $s_k$, $c_k$, and $g_k$.  
In practice, $x_k$, $s_k$, $c_k$, $g_k$ are implemented as tuple of vectors of the FP formats used to perform evaluations (*i.e.* `FPMPNLPModel.FPList`).

**Example 1: Simple Callback**
Here is a simple callback function for computing the objective function that does not take evaluation error into account.
```julia
function my_compute_f_at_c!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T) where {H, L, E, T <: Tuple}
  st.f⁺ = objmp(m, c[π.πc]) # FP format m.FPList[π.πc] will be used for obj evaluation.
end
```

The implementation of MPR2 automatically updates the containers for x, s, c, and g during execution, so that the user does not have to deal with that part. The function `MultiPrecisionR2.umpt!()` (see documentation) update the containers.

**Warning:**
`MultiPrecisionR2.umpt!(x::Tuple, y::Vector{S})` updates only the vectors of x of FP format with precision greater or equal to the FP format of y. `umpt!` is implemented this to avoid rounding error due to casting into a lower precision format and overflow. This is closely related to the concept of "forbidden evaluation" detailed in the [next section](#forbidden-evaluations).
If the precision `π.πx` = 2, it means that the `x[i]`s vectors are up-to-date for i>=2. 

**Example: Container Update**
```julia
using MultiPrecisionR2
FP = [Float16,Float32,Float64]
xini = ones(5)
x = Tuple(fp.(xini) for fp in FP)
xnew = Float32.(zeros(5))
umpt!(x,xnew)
x # only x[2] and x[3] are updated
```

**Example: Lower Precision Casting Error**
```julia
x64 = [70000.0,1.000000001,0.000000001,-230000.0] # Vector{Float64}
x16 = Float16.(x64) # definitely not x64
```

### **Forbidden Evaluations**
A "forbidden evaluation" consists in evaluating the objective or the gradient with a FP format lower than the FP format of the point where it is evaluated. For example, if `x` is a `Vector{Float64}`, evaluating the objective in Float16 will implicitely cast `x` into a `Vector{Float16}` and then evaluate the objective with this Float16 vector. The problem is that due to rounding error, the casted vector is different than the initial Float64 `x`. The objective is therefore not evaluated at `x` but at a different point. This causes numerical instability in MPR2 and must be avoided.

**Example: Forbidden Evaluation**
```julia
f(x) = 1/(x-10000) # objective
x64 = 10001.0
f(x64) # returns 1 as expected
x16 = Float16(x64) # rounding error occurs: x16 != x64
f(x16) # definitely not the expected value for f(x)
```

When implementing the callback functions, the user should make sure that the evaluation precision selected for objective or gradient evaluation (`π.πf`, `π.πg`) is always greater or equal than the precision of the point where the evaluation is performed (`π.πx` or `π.πc`).

### **Step and Candidate Computation Precision**
MPR2 implementation checks for possible under/overflow when computing the step `s` and the candidate `c` and increase FP format precision if necessary to avoid that. MPR2 implementation also updates `π.πs` and `π.πc` if necessary. These careful step and candidate computations are implemented in the `MultiPrecisionR2.ComputeStep!()` and `MultiPrecisionR2.computeCandidate!()` (see documentation).
It is important to note that, even if the gradient `g` has been computed with `π.πg` precision, the precision `π.πs` and `π.πc` can be greater than `π.πg` because overflow has occurred. Has a consequence, the user should not take for granted than `π.πc` == `π.πg` when selecting FP format for objective/gradient evaluation at `c` but must rely on the candidate precision `π.πc`.

**Example: Step Overflow**
```julia
g(x) = 2*x # gradient
x16 = Float16(1000)
sigma = Float16(1/2^10) # regularization parameter
g16 = g(x16)
s = g16/Float16(sigma) # overflow
s = Float32(g16)/Float32(sigma) # no overflow, s is a Float32, this is what computeStep!() does
```

### **Candidate Precision Selection**
In light of the "forbidden evaluation" concept (see section [Forbidden Evaluations](#forbidden-evaluations)), if no particular care is given, the objective/gradient evaluation precisions can only increase from one iteration to another. To illustrate that consider that 
1. At iteration $k$: $x_k$ is Float32, meaning that $\hat{g}(x_k)$ is Float32 (or higher precision FP format), but let's say Float32 here. By extension, $s_k = -g_k/\sigma_k$ , and $c_k = x_k+s_k$ are both Float32. It means that $f(x_k)$ and $f(x_k+s_k)$ are computed with Float32 or higher because of forbidden evaluations.
2. At iteration $k+1$:
    * If the iteration is successful: then $x_{k+1} = c_k$ is Float32, meaning again that $\hat{g}(x_{k+1})$ is Float32 or higher precision format, and $s_{k+1}$ and $c_{k+1}$ are also Float32 or higher precision format. It further means that $f(x_{k+1})$ and $f(c_{k+1})$ can only be computed with Float32 or higher percision format.
    * If the iteration is unsuccessful: then $x_{k+1} = x_k$ is Float32 and we have the same than if the iteration is successful.

To overcome this issue, and enable to decrease the FP formats used for objective/gradient evaluation, the user has the freedom to chose at iteration $k$ the FP format of $c_{k+1}$. Indeed, there is no restriction on how the candidate is computed. Considering the above example, at iteration $k+1$ $c_{k+1}$ is Float32 but we can cast it (with rounding error) into a Float16, without breaking the convergence. Casting $c_{k+1}$ allows to compute $f(c_{k+1})$ with Float16, and if the iteration is successful, $x_{k+2} = c_{k+1}$ is also a Float16, meaning that the gradient and objective can be computed with Float16 at $x_{k+2}$.

The callback function `selectPif!()`, called at the end of the main loop (see section [Minimal Implementation Description](#minimal-implementation-description)) update `π.πc` so that `MPnlp.FPList[π.πc]` will be the FP format of `c` at the next iteration. The function `ComputeCandidate!()`, at the begining of the main loop, handles the casting of the candidate into `MPnlp.FPList[π.πc]`.

The expected template for `selectPif!()` callback function is `function selectPic!(π::MPR2Precisions)`. Only `π.πc` is expected to be modify.

|Callback| Modified Variables|
| ------ | ----------------- |
|`selectPic!()`| `π.πc`

## What `MultiPrecisionR2.solve!()` Handles
* Runs the main loop until a stopping condition is reached (max iteration or $\|\nabla f(x)\| \leq \epsilon$).
* Uses the default callback functions for whichever has not been provided by the user.
* Deals with error due to norm computation to ensure $\|\nabla f(x)\| \leq \epsilon$.
* Deals with high precision format computation to simulate "exactly computed" values (see sections [High Precision Format](#high-precision-format) and [High Precision Format: MPR2 Solver](#high-precision-format-mpr2-solver)).
* Update the containers `x`, `s`, `c` and `g` (see [Multi-Precision Evaluation and Vector Containers](#multi-precision-evaluation-and-vector-containers))
* Make sure no under/overflow occurs when computing `s` and `c`, update the precision `π.πs` and `π.πc` if necessary (see [Step and Candidate Computation Precision](#step-and-candidate-computation-precision))

## What `MultiPrecisionR2.solve!()` Does not Handle
* `solve!()` does not check that objective/gradient evaluation are preformed with a suitable FP format in the callback functions(see sections [Multi-Precision Evaluation and Vector Containers](#multi-precision-evaluation-and-vector-containers) and [Forbidden Evaluation](#forbidden-evaluations)).
* `solve!()` does not ensure convergence if the user uses its own callback functions.
* `solve!()`does not update objective/gradient values and precision outside the callback functions. See [Callback Functions: Expected Behavior](#callback-functions-expected-behavior) for proper callbacks implementation.
* `solve!()` does not handle overflow that might occur when evaluation the objective/gradient in the callback functions. It is up to the user to make sure no overflowed value is returned by the callbacks. See [Implementation Examples](#implementation-examples) for dealing with overflow properly.
* `solve!()` does not throw error/warning if evaluations in callback have failed. It is up to the user to handle that.

## Callback Functions Cheat Sheet

|Callback| Description | Outputs | Expected Modified Variables |
| ------ | ----------- | ------- | --------------------------- | 
|`compute_f_at_x!()`| Select obj. FP format, compute $f(x_k)$ and $ωf(x_k)$ |prec_fail::Bool : `true` if $\omega_f(x_k)$ is too big, stops main loop| `st.status`, `st.f`, `st.ωf`, `π.πf`
|`compute_f_at_c!()`|Select obj. FP format and compute $f(c_k)$ and $ωf(c_k)$ | prec_fail::Bool: `true` if $\omega_f(c_k)$ is too big, stops main loop|  `st.status`, `st.f⁺`, `st.ωf⁺`, `π.πf⁺`
|`compute_g!()`| Select grad FP format and compute $g(c_k)$ and $ωg(c_k)$ | prec_fail::Bool: `true` if $\omega_g(c_k)$ is too big, stops main loop| `st.status`, `g`, `st.ωg`, `π.πg` 
|`recompute_g()`| Select grad FP format and recompute $g(x_k)$ and $ωg(x_k)$ | prec_fail::Bool: `true` if $\omega_g(c_k)$ is too big, stops main loop, g_recompute::Bool: `true` if $\hat{g}(x_k)$ was recomputed|`st.status`, `g`, `st.ωg`, `π.πg`, `st.ΔT`, `π.ΔT`, `st.x_norm`, `π.πnx`, `st.s_norm`, `π.πns`, `st.ϕ`, `st.ϕhat`, `π.πc`, `st.μ` 
|`selectPic!()`| | void | `π.πc` 



## Implementation Examples

### Example 1: Precision Selection Strategy Based on Step Size (Error Free)
This example implements a precision selection strategy for the objective and gradient based on the norm of the step size, which does not take into account evaluation errors. The strategy is to choose the FP format for evaluation such that the norm of the step is greater than the square root of the unit roundoff.

The callback functions must handles precision selection for evaluations and optionally error/warning messages if evaluation fails (typically overflow or lack of precision)

```julia
using LinearAlgebra

function my_compute_f_at_c!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T) where {H, L, E, T <: Tuple}
  πmax = length(m.FPList) # get maximal allowed precision
  eval_prec = findfirst(u -> sqrt(u) < st.s_norm, m.UList) # select precision according to the criterion
  if eval_prec === nothing # not enough precsion
    @warn " not enough precision for objective evaluation at c: ||s|| = $(st.s_norm) < sqrt(u($(m.FPList[end]))) = $(sqrt(m.UList[end]))"
    st.status = :exception
    return true
  end
  π.πf⁺ = max(eval_prec,π.πc) # evaluation precision should be greater or equal to the FP format of the candidate (see forbidden evaluation)
  st.f⁺ = obj(m,c[π.πf⁺]) # eval objective only. solve!() made sure c[π.πf⁺] is up-to-date (see containers section)
  while st.f⁺ == Inf # check for overflow
    π.πf⁺ += 1
    if π.πf⁺ > πmax
      @warn " not enough precision for objective evaluation at c: overflow"
      st.status = :exception
      return true # objective overflow with highest precision FP format: this is a fail
    end
    st.f⁺ = obj(m,c[π.πf⁺])
  end
  return false
end
  
function my_compute_f_at_x!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, x::T) where {H, L, E, T <: Tuple}
  πmax = length(m.FPList) # get maximal allowed precision
  if st.iter == 0 # initial evaluation, step = 0, choose max precision
    π.πf = πmax
  else # evaluation in main loop
    eval_prec = findfirst(u -> sqrt(u) < st.s_norm, m.UList) # select precision according to the criterion
    if eval_prec === nothing # not enough precsion
      @warn " not enough precision for objective evaluation at x: ||s|| = $(st.s_norm) < sqrt(u($(m.FPList[end]))) = $(sqrt(m.UList[end]))"
      st.status = :exception
      return true
    end
    π.πf = max(eval_prec,π.πx) # evaluation precision should be greater or equal to the FP format of the current solution (see forbidden evaluation)
  end
  st.f = obj(m,x[π.πf]) # eval objective only. solve!() made sure x[π.πf] is up-to-date (see containers section)
  while st.f == Inf # check for overflow
    π.πf += 1
    if π.πf > πmax
      @warn " not enough precision for objective evaluation at x: overflow"
      st.status = :exception
      return true # objective overflow with highest precision FP format: this is a fail
    end
    st.f = obj(m,x[π.πf])
  end
  return false
end

function my_compute_g!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T, g::T) where {H, L, E, T <: Tuple}
  πmax = length(m.FPList) # get maximal allowed precision
  if st.iter == 0 # initial evaluation, step = 0, choose max precision
    π.πg = πmax
  else # evaluation in main loop
    eval_prec = findfirst(u -> sqrt(u) < st.s_norm, m.UList) # select precision according to the criterion
    if eval_prec === nothing # not enough precsion
      @warn " not enough precision for gradient evaluation at c: ||s|| = $(st.s_norm) < sqrt(u($(m.FPList[end]))) = $(sqrt(m.UList[end]))"
      st.status = :exception
      return true
    end
    π.πg = max(eval_prec,π.πg) # evaluation precision should be greater or equal to the FP format of the candidate (see forbidden evaluation)
  end
  grad!(m,c[π.πg],g[π.πg]) # eval gradient only. solve!() made sure x[π.πg] is up-to-date (see containers section)
  while findfirst(elem->elem == Inf,g[π.πg]) !== nothing # check for overflow, gradient vector version
    π.πg += 1
    if π.πg > πmax
      @warn " not enough precision for gradient evaluation at c: overflow"
      st.status = :exception
      return true # objective overflow with highest precision FP format: this is a fail
    end
    grad!(m,c[π.πg])
  end
  return false
end

function my_recompute_g!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, x::T, g::T, s::T) where {H, L, E, T <: Tuple}
  # simply update norm of the step, since recompute_g!() is called at the begining of the main loop after step computation
  πmax = length(m.FPList) # get maximum precision index
  π.πns = π.πs # select precision for step norm computation
  s_norm = norm(s[π.πs])
  while s_norm == Inf || s_norm ==0.0 # handle possible over/underflow
    π.πns = π.πns+1 # increase precision to avoid over/underflow
    if π.πns > πmax
      st.status = :exception
      return true, false # overflow occurs with max precion: cannot compute s_norm with provided FP formats. Returns fail.
    end
    s_norm = norm(s[π.πns]) # compute norm with higher precision step, solve!() made sure s[π.πns] is up-to-date
  end
  st.s_norm = s_norm
  return false, false
end
```

Let's try this implementation on a simple quadratic objective.

```julia
FP = [Float16, Float32] # selected FP formats,
f(x) = x[1]^2 + x[2]^2 # objective function
omega = [0.0,0.0]
x = Float32.(1.5*ones(2)) # initial point
MPmodel = FPMPNLPModel(f,x,FP, ωfRelErr = omega, ωgRelErr = omega); # indicates the use of relative error only to avoid interval evaluation, relative errors will not be computed with above callbacks
stat = MPR2(MPmodel;
compute_f_at_x! = my_compute_f_at_x!,
compute_f_at_c! = my_compute_f_at_c!,
compute_g! = my_compute_g!,
recompute_g! = my_recompute_g!);
stat  # first-order stationary point has been found
```

Let's now try our implementation on the Rosenbrock function.

```julia
FP = [Float16, Float32] # selected FP formats,
f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2 # Rosenbrock function
omega = [0.0,0.0]
x = Float32.(1.5*ones(2)) # initial point
MPmodel = FPMPNLPModel(f,x,FP, ωfRelErr = omega, ωgRelErr = omega); # indicates the use of relative error only to avoid interval evaluation, relative errors will not be computed with above callbacks
stat = MPR2(MPmodel;
compute_f_at_x! = my_compute_f_at_x!,
compute_f_at_c! = my_compute_f_at_c!,
compute_g! = my_compute_g!,
recompute_g! = my_recompute_g!); # throw lack of precision warning
```

The strategy implemented for precision selection does not allow to find a first-order critical point for the Rosenbrock function: the step becomes too small before MPR2 converges. Although this implementation is fast since it does not bother with evaluation errors, it is not very satisfactory since Example 1 in section [Lack of Precision](#lack-of-precision) shows that the default implementation is able to converge to a first-order critical point.
This highlights that it is important to understand how rounding errors occur and affect the convergence of the algorithm (see section [MPR2 Algorithm: General Description](#mpr2-algorithm-general-description)) and that "naive" strategies like the one implemented above might not be satisfactory.

### Example 2: Switching to Gradient Descent When Lacking Objective Precision
It might happen that `solve!()` stops early because the objective evaluation lacks precision. Consider for example that we use consider relative evaluation error for the objective. If MPR2 converges to a point where the objective is big, the error can be big too, and if the gradient is small the convergence condition $\omega f(x_k) \leq \eta_0 \Delta T_k = \|\hat{g}(x_k)\|^2/\sigma_k$ is likely to fail. In that case, the user might want to continue running the algorithm without caring about the objective, that is, as a simple gradient descent.
`solve!()` implementation allows enough flexibility to do so. In the implementation below, the user defined structure `e` is used to indicate what "mode" the algorithm is running: default mode or gradient descent. The callbacks `compute_f_at_x!` sets `st.f = Inf` and `compute_f_at_c!` sets `st.f⁺ = 0` if gradient descent mode is used. This ensures that $\rho_k = Inf \geq \eta_1$ and the step is accepted in gradient descent mode.
In the implementation below, `compute_f_at_x!` and `compute_f_at_c!` selects the precision such that $\omega f(x_k) \leq \eta_0 \Delta T_k$ in default mode. We implement `compute_g!` to set `σ` so that `ComputeStep!()` will use the learning rate `1/σ`. We use the default `recompute_g` callback.

```julia
mutable struct my_struct
  gdmode::Bool
  learning_rate
end

function my_compute_f_at_c!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T) where {H, L, E, T <: Tuple}
  if !e.gdmode
    ωfBound = p.η₀*st.ΔT
    π.πf⁺ = π.πx # basic precision selection strategy
    st.f⁺, st.ωf⁺, π.πf⁺ = objReachPrec(m, c, ωfBound, π = π.πf⁺)
    if st.f⁺ == Inf # stop algo if objective overflow
      @warn "Objective evaluation overflow at x"
      st.status = :exception
      return true
    end
    if st.ωf⁺ > ωfBound # evaluation error too big
      @warn "Objective evaluation error at x too big to ensure convergence: switching to gradient descent"
      e.gdmode = true
      st.f⁺ = 0
      return false
    end
  else # gradient descent mode
    st.f⁺ = 0
  end
  return false
end

function my_compute_f_at_x!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, p::MPR2Params{H, L}, e::E, x::T) where {H, L, E, T <: Tuple}
  πmax = length(m.EpsList)
  if st.iter == 0 # initial evaluation before main loop
    st.f, st.ωf, π.πf = objReachPrec(m, x, m.OFList[end], π = π.πf)
  else # evaluation in the main loop
    if !e.gdmode
      ωfBound = p.η₀*st.ΔT
      if st.ωf > ωfBound # need to reevaluate the objective at x
        if π.πf == πmax # already at highest precision 
          @warn "Objective evaluation error at x too big to ensure convergence: switching to gradient descent"
          e.gdmode = true
          st.f = Inf
          return false
        end
        π.πf += 1 # increase evaluation precision of f at x
        st.f, st.ωf, π.πf = objReachPrec(m, x, ωfBound, π = π.πf)
        if st.ωf > ωfBound # error evaluation too big with max precison
          @warn "Objective evaluation error at x too big to ensure convergence: switching to gradient descent"
          e.gdmode = true
          st.f = Inf
          return false
        end
      end
    else # gradient descent mode
      st.f = Inf
    end
  end
  return false
end

function my_compute_g!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T, g::T) where {H, L, E, T <: Tuple}
  π.πg = π.πc # default strategy, could be a callback
  st.ωg, π.πg = gradReachPrec!(m, c, g, m.OFList[end], π = π.πg)
  if e.gdmode
    st.σ = 1/e.learning_rate
  end
  return false
end
```
Let us first run `MPR2()` with the default implementation and relative evaluation error.
```julia
FP = [Float16, Float32] # selected FP formats,
#f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2 # Rosenbrock function
f(x) = x[1]^2 + x[2]^2 +0.5
omegaf = Float64.([0.01,0.005])
omegag = Float64.([0.05,0.01])
x = Float32.(1.5*ones(2)) # initial point
MPmodel = FPMPNLPModel(f,x,FP, ωfRelErr = omegaf, ωgRelErr = omegag);
stat = MPR2(MPmodel,verbose=1); # stops at iteration 3, throw lack of precision warning
```

We run `MPR2()` with the callback functions defined above and the default callbacks for `compute_g!()` and `recompute_g!()`. We use relative objective and gradient error.
```julia
FP = [Float16, Float32] # selected FP formats,
#f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2 # Rosenbrock function
f(x) = x[1]^2 + x[2]^2 +0.5
omegaf = Float64.([0.01,0.005])
omegag = Float64.([0.05,0.01])
x = ones(Float32,2) # initial point
MPmodel = FPMPNLPModel(f,x,FP, ωfRelErr = omegaf, ωgRelErr = omegag);
e = my_struct(false,1e-2)
stat = MPR2(MPmodel;
e = e,
compute_f_at_x! = my_compute_f_at_x!,
compute_f_at_c! = my_compute_f_at_c!,
compute_g! = my_compute_g!); # switch to gradient descent at iteration 3, converges to first order critical point
```