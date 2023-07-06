```@meta
CurrentModule = MultiPrecisionR2
```

# MultiPrecisionR2

`MultiPrecisionR2.jl` implements MPR2, a multi precision extension of the Quadratic Regularization (R2) algorithm, for solving continuous unconstrained non-convex problems with several Floating Point (FP) formats, for example Float16, Float32 and Float64. MPR2 aims for evaluating the objective and the gradient with the lowest precision FP format to save computational time and energy while controlling computation error to maintain convergence.

`MultiPrecisionR2.jl` implements MPR2 in such a way that it allows the user some flexibility. By default, rounding errors and precision selection for the objective function and the gradient evaluation are handled such that the convergence is guaranteed (easy use, see [Basic Use](#basic-use) chapter). The user can choose to implement its own strategies for precision selection via callback functions (see [Advanced Use](#advanced-use) chapter), but in that case it is up to the user to ensure the convergence of the algorithm. 

# Table of Contents
1. [Quick Start](#quick-start)
2. [Motivation](#motivation)
3. [MPR2 Algorithm: General Description](#mpr2-algorithm-general-description)  
    1. [Notations](#notations)  
    2. [MPR2 Algorithm Broad Description](#mpr2-algorithm-broad-description-differs-from-package-implementation)  
    3. [Rounding Error Handling](#rounding-errors-handling)  
    4. [Conditions on Parameters](#conditions-on-parameters)  
4. [Basic Use](#basic-use)
    1. [FPMPNLPModels: Creating a Multi-Precision Model](#fpmpnlpmodel-creating-a-multi-precision-model)
        1. [Evaluation Mode](#evaluation-mode)
        2. [High Precision Format](#high-precision-format)
        3. [Gradient and Dot Product Error: Gamma Callback Function](#gradient-and-dot-product-error-gamma-callback-function)
    2. [MPR2Solver](#mpr2solver)
        1. [Gamma Function](#gamma-function)
        2. [High Precsion Format: MPR2 Solver](#high-precision-format-mpr2-solver)
        3. [Lack of Precision](#lack-of-precision)
5. [Advanced Use](#advanced-use)
    1. [Diving into MPR2 Implementation](#diving-into-mpr2-implementation)
        1. [Minimal Implementation Description](#minimal-implementation-description)
        2. [Callback Functions: Templates](#callback-functions-templates)
        3. [Callback Functions: Expected Behaviors](#callback-functions-expected-behaviors)
        4. [Multi-Precision Evaluation and Vectors Containers](#multi-precision-evaluation-and-vectors-containers)
        5. [Forbidden Evaluations](#forbidden-evaluations)
        6. [Step and Candidate Computation Precision](#step-and-candidate-computation-precision)
        7. [Candidate Precision Selection](#candidate-precision-selection)
    2. [What MPR2Solver Handles](#what-mpr2solver-handles)
    3. [What MPR2Solver Does not Handle](#what-mpr2solver-does-not-handle)
    4. [Implementation Examples](#implementation-examples)

# Quick Start
```julia
using MultiPrecisionR2
using ADNLPModels
using IntervalArithmetic

setrounding(Interval,:accurate)
FP = [Float16,Float32] # define floating point formats used by the algorithm for objective and gradient evaluation
f(x) = x[1]^2 + x[2]^2 # some objective function
x0 = ones(Float32,2) # initial point
mpmodel = FPMPNLPModel(f,x0,FP); # instanciate a Floating Point Multi Precision NLPModel (FPMPNLPModel)
stat = MPR2(mpmodel) # run the algorithm
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
stat = MPR2(mpmodel) # run the algorithm
```

**Warnings**
1. MultiPrecisionR2.jl works only with Floating Point formats (BFloat16,Float16,Float32,Float64,Float128)
2. Unfortunately, other modes than `:accurate` for interval evaluation are not guaranteed to work with FP formats different from Float32 and Float64. 
3. If interval evaluation mode is used, interval evaluation for the objective and the gradient is automatically tested upon FPMPNLPModel instanciation.  An error is thrown if the evaluation fails. This might happen for several reasons related to [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl/blob/master/README.md) package.

# Motivation
Here is the comparison between R2 algorithm from JSOSolvers.jl run with Float64 and MPR2 from MultiPrecisionR2.jl run with Float16, Float32 and Float64 over a set of unconstrained problems from OptimizationProblems.jl. For a fair comparison, we set MPR2 evaluation errors for objective function and gradient to zero with FLoat64.
The time and energy savings offer by MPR2 compared to R2 is estimated with the rules of thumb:

Number of bits divided by two -> time computation divided by two and energy consumption divided by four.


```julia
using MultiPrecisionR2
using NLPModels
using ADNLPModels
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using JSOSolvers

FP = [Float16,Float32,Float64] # MPR2 Floating Point formats
omega = Float64.([sqrt(eps(Float16)),sqrt(eps(Float32)),0]) # MPR2 relative errors, computations assumed exact with Float64
r2_obj_eval = [0]
r2_grad_eval = [0]
mpr2_obj_eval = zeros(Float64,length(FP))
mpr2_grad_eval = zeros(Float64,length(FP))
nvar = 100 #problem dimension (if scalable)
max_iter = 1000

meta = OptimizationProblems.meta
names_pb_vars = meta[(meta.has_bounds .== false) .& (meta.ncon .== 0), [:nvar, :name]] #select unconstrained problems
for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(n=$nvar,type=Val(Float64),gradient_backend = ADNLPModels.GenericForwardDiffADGradient)"))
  mpmodel = FPMPNLPModel(nlp,FP,ωfRelErr=omega,ωgRelErr=omega);
  statr2 = R2(nlp,max_eval=max_iter)
  r2_obj_eval .+= nlp.counters.neval_obj
  r2_grad_eval .+= nlp.counters.neval_grad
  statmpr2 = MPR2(mpmodel,max_iter = max_iter)
  mpr2_obj_eval .+= [haskey(mpmodel.counters.neval_obj,fp) ? mpmodel.counters.neval_obj[fp] : 0 for fp in FP]
  mpr2_grad_eval .+= [haskey(mpmodel.counters.neval_grad,fp) ? mpmodel.counters.neval_grad[fp] : 0 for fp in FP]
end
mpr2_obj_time = sum(mpr2_obj_eval.*[1/4,1/2,1])
obj_time_save = mpr2_obj_time/r2_obj_eval[1]
mpr2_obj_energy = sum(mpr2_obj_eval.*[1/16,1/4,1])
obj_energy_save = mpr2_obj_energy/r2_obj_eval[1]
mpr2_grad_time = sum(mpr2_grad_eval.*[1/4,1/2,1])
grad_time_save = mpr2_grad_time/r2_grad_eval[1]
mpr2_grad_energy = sum(mpr2_grad_eval.*[1/16,1/4,1])
grad_energy_save = mpr2_grad_energy/r2_grad_eval[1]
println("Possible time saving for objective evaluation with MPR2: $(round((1-obj_time_save)*100,digits=1)) %")
println("Possible time saving for gradient evaluation with MPR2: $(round((1-grad_time_save)*100,digits=1)) %")
println("Possible energy saving for objective evaluation with MPR2: $(round((1-obj_energy_save)*100,digits=1)) %")
println("Possible energy saving for gradient evaluation with MPR2: $(round((1-grad_energy_save)*100,digits=1)) %")
```



# MPR2 Algorithm: General Description 

## **Notations**
* $fl$: finite-precision computation
* $u$: unit round-off for a given FP format
* $\delta$: rounding error induced by one FP operation ($+,-,/,*$), for example $fl(x+y) = (x+y)(1+\delta)$. Bounded as $|\delta|\leq u$.
* $\pi$: FP format index, also called **precision**
* $f: x \rightarrow f(x)$: objective function
* $\hat{f}: x,\pi_f \rightarrow fl(f(x,\pi_f))$: finite precision counterpart of $f$ with FP format corresponding to index $\pi_f$  
* $\nabla f: x \rightarrow \nabla f(x)$: gradient of $f$
* $\hat{g}: x, \pi_g \rightarrow fl(\nabla f(x),\pi_g)$ : finite precision counterpart of $\nabla f $ with FP format corresponding to index $\pi_g$
* $\omega_f(x)$: bound on finite precision evaluation error of $f$, $| \hat{f}(x,\pi_f) -f(x) | \leq \omega_f(x)$
* $\omega_g(x)$ : bound on finite precision evaluation error of $\nabla f$, $\| \hat{g}(x,\pi_g) -\nabla f(x) \| \leq \omega_g(x)\|\hat{g}(x,\pi_g)\|$

## **MPR2 Algorithm Broad Description** (differs from package implementation)
MPR2 is described by the following algorithm. Note that the actual implementation in the package differs slightly. For an overview of the actual implementation, see section [Diving Into MPR2 Implementation](#diving-into-mpr2-implementation).

In the algorithm, *compute* means compute with finite-precision machine computation, and *define* means "compute exactly", *i.e* with infinite precision. Defining a value is therefore not possible to perform on a (finite-precision) machine. This point is discussed later. 

**Inputs**: 
  * Initial values: $x_0$, $\sigma_0$
  * Tunable Parameters: $0 < \eta_0 < \eta_1 < \eta_2 < 1$, $0 < \gamma_1 < 1 < \gamma_2$, $\kappa_m$
  * Gradient tolerance: $\epsilon$
  * List of FP formats (*e.g* [Float16, Float32, Float64])

**Outputs**
  * $x_k$ such that $\nabla f(x_k) \leq \epsilon \|\nabla f(x_0)\|$

**Initialization**
1. Compute $f_0 = \hat{f}(x_0,\pi_f)$
2. Compute $g_0 = \hat{g}(x_0,\pi_g)$

**While** $\|g_k\| > \dfrac{\epsilon}{1+\omega_g(x_k)}$ **do**

3. Compute $s_k = -g_k/\sigma_k$
4. Compute $c_k = x_k + s_k$
5. Compute $\Delta T_k = g_k^Ts_k$ # model reduction
6. Define $\mu_k$ # gradient error indicator

* **If** $\mu_k \geq \kappa_m$

  7. Select precisions such that $\mu_k \leq \kappa_m$, go to 3.

* **End If**

* **If** $\omega_f(x_k) > \eta_0 \Delta T_k$

  8. Select $\pi_f$ such that $\omega_f(x_k) \leq \eta_0 \Delta T_k$
  9. Compute $f_k = f(x_k,\pi_f)$

* **End If**

10. Select $\pi_f^+$ such that $\omega(c_k) \leq \eta_0 \Delta T_k$
11. Compute $f_k^+ = \hat{f}(c_k,\pi_f^+)$
12. Compute $\rho_k = \dfrac{f_k - f_k^+}{\Delta T_k}$

* **If** $\rho_k \geq \eta_1$ # step acceptance
  
  13. $x_{k+1} = c_k$, $f_{k+1} = f_k^+$

* **End If**

14. Compute $ \sigma_k = \left\{ \begin{array}{lll}
  \gamma_1 \sigma_k & \text{if} & \rho_k \geq \eta_2 \\
    \sigma_k & \text{if} & \rho_k \in  \eta_1,\eta_2  \\
    \gamma_2 \sigma_k & \text{if} & \rho_k < \eta_1
  \end{array}
   \right.$

**End While**  
15. return $x_k$

MPR2 stops either when:
1. a point $x_k$ satisfying $\|g_k\| \leq \dfrac{\epsilon}{1+\omega_g(x_k)}$ has been found (see [$\omega_g$ definition](#notations)), which ensures that $ \|\nabla f(x_k)\| \leq \epsilon$, or 
2. no FP format enables to achieve required precision on the objective or $\mu$ indicator.
The indicator **$\mu_k$** aggregates finite-precision errors due to gradient evaluation ($\omega_g(x_k)$), and the computation of the step ($s_k$), candidate ($c_k$) and model reduction $\Delta T_k$ as detailed in the [**Rounding Error Handling**](#rounding-error-handling) section.     

## **Rounding Errors Handling**
Anything computed in MPR2 suffers from rounding errors since it runs on a machine, which necessarily performs computation with finite-precision arithmetic. Below is the list of rounding errors that are handled by MPR2 such that convergence and numerical stability is guaranteed.
1. **Dot Product Error: $\gamma_n$**
  The model for the dot product error that is used by default is  
  $|fl(x.y) - x.y| \leq |x|.|y| \gamma(n,u),$ with  
  $\gamma: n,u \rightarrow n*u$  
  where $x$ and $y$ are two FP vectors (same FP format) of dimension $n$ and $u$ is the round-off unit of the FP format. This is a crude yet guaranteed upper bound on the error. The user can choose its own formula for $\gamma$, see [FPMPNLPModel: Creating a Multi-Precision Model](#fpmpnlpmodel-creating-a-multi-precision-model) section for practical use.
2. **Candidate Computation**
   Both the step and the candidate are computed inexactly.
   * Inexact Step: the computed step is $s_k = fl(g_k/\sigma_k) = (g_k/\sigma_k)(1+\delta) \neq g_k/\sigma_k$.
   * Inexact Candidate: the computed candidate is $c_k = fl(x_k+s_k) = (x_k+s_k)(1+\delta) \neq x_k+s_k$.
     
   These two errors implies that $c_k$ is not along the descent direction $g_k$ and as such can be interpreted as additional gradient error to  $\omega_g(x_k)$. To handle these errors, MPR2 computes the ratio $\phi_k = \|x_k\|/||s_k||$. The greater $\phi_k$, the greater the possible deviation of $c_k$ from the direction $g_k$. These errors are aggregated in the $\mu_k$ indicator (detailed in 5.).
   
3. **Model Decrease Computation**
  The FP computation error for the model decrease $\Delta T_k$ is such that,  
  $fl(\Delta T_k) =  \Delta T_k (1+\vartheta_n)$, with $|\vartheta_n| \leq \gamma(n,u)$  
  MPR2 handles this error by aggregating $\alpha(n,u)$ in the indicator $\mu_k$, with $\alpha(n,u) = \dfrac{1}{1-\gamma(n,u)}$

4. **Norm Computation**  
  The finite-precision error for norm computation of a FP vector $x$ of dimension $n$ is  
  $ |fl(\|x\|)-\|x\|| \leq fl(\|x\|)\beta(n+2,u)$ with  
  $\beta:n,u \rightarrow \max(|\sqrt{\gamma(n,u)-1}-1|,|\sqrt{\gamma(n,u)+1}-1|)$  
  $\beta$ function cannot be chosen by the user.  
  The $\beta$ function is used in MPR2 implementation to handle finite-precision error of norm computation for 
  * $\|g_k\|$ norm computation: the stopping criterion implemented in MPR2 is  
    $\|g_k\| \leq \dfrac{1}{1+\beta (n+2,u)}\dfrac{1}{1+\omega_g(x_k)}$ which ensures that $\|\nabla f(x_k)\|\leq \epsilon$. 
  * $\phi_k$ ratio computation (see 3.): $\phi_k$ requires the norm of $x_k$ and $s_k$. MPR2 actually defines 
  $\phi_k =  \dfrac{fl(\|x_k\|)}{fl(\|x_k\|)}\dfrac{1+\beta(n+2,u)}{1-\beta(n+2,u)}$ which ensures that $\phi_k \geq \|x_k\| / \|s_k\|$.

5. **Mu Indicator**  
  $\mu_k$ is an indicator which aggregates
  * the gradient evaluation error $\omega_g(x_k)$,
  * the inexact step and candidate errors ($\phi_k$),
  * the model decrease error ($\gamma(n+1,u)$, $\alpha(n,u)$).  
  The formula for $\mu_k$ is  
  $\mu_{k} = \dfrac{\alpha(n,u) \omega_g(x_k)(1+u(\phi_k +1)) + \alpha(n,u) \lambda_k + u+ \gamma(n+1,u)\alpha(n+1,u)}{1-u}$.  
  Note that if no rounding error occurs for ($u = 0$), one simply has $\mu_k = \omega_g(x_k)$.  
  The implementation of line 7. of MPR2 (as described in the [above section](#mpr2-algorithm-broad-description-differs-from-package-implementation)) consists in recomputing the step, candidate, $x_k$ and/or $s_k$ norm, model decrease or gradient with higher precision FP formats (therefore decreasing $u$) until $\mu_k \leq \kappa_m$. For details about default strategy, see `recomputeMu!()` documentation.

## Conditions on Parameters
  MPR2 parameters can be chosen by the user (see Section [Basic Use](#basic-use)) but must satisfy the following inequalities:
  * $0 \leq \eta_0 \leq \frac{1}{2}\eta_1$
  * $0 \leq \eta_1 \leq \eta_2 < 1$
  * $\eta_0+\dfrac{\kappa_m}{2} \leq 0.5(1-\eta_2)$
  * $\eta_2 < 1$>
  * $0<\gamma_1<1<\gamma_2$


# Basic Use

MPR2 solver relies on multi-precision models structure `FPMPNLPModels` (Floating Point Multi Precision Non Linear Programming Models), that derives from [`NLPModels.jl`](). This structure embeds the problem and provides the functions to evaluate the objective and the gradient with several FP formats. These evaluation functions are used by `MPR2Solver` to evaluate the objective and gradient with different FP format.

## **FPMPNLPModel**: Creating a Multi-Precision Model
See `FPMPNLPModel` documentation.

### **Evaluation Mode**

`FPMPNLPModel` implements tools to evaluate the objective and the gradient and the error bounds $\omega_f(x_k)$ and $\omega_g(x_k)$ with two modes:
1. **Interval Arithmetic**: Using interval arithmetic enables to track rounding errors and to provide guaranteed bounds.`MultiPrecisionR2.jl` relies on [`IntervalArithmetic.jl`](https://juliaintervals.github.io/pages/packages/intervalarithmetic/) library to perform interval evaluations of the gradient and the objective.  
  **Warning**: Using interval evaluation can lead to significantly long computation time. Interval evaluation might fail for some objective functions due to `IntervalArithmetic.jl` errors.
2. **Relative Error**: The error bounds $\omega_f(x_k)$ and $\omega_g(x_k)$ are **estimated** as fractions of $|\hat{f}(x_k)|$ and $\|\hat{g}(x_k)\|$, respectively. It is to the user to provide these fraction values.  
By default, interval arithmetic is used for error bound computation. If relative error bounds have to be used, `ωfRelErr` and `ωgRelErr` keyword arguments must be used when calling the constructor function.

**FPMPNLPModel Example 1: Interval Arithmetic for error bounds**
```julia
using MultiPrecisionR2
using IntervalArithmetic
using ADNLPModels

setrounding(Interval,:accurate)
FP = [Float16, Float32] # selected FP formats
f(x) = x[1]^2 + x[2]^2 # objective function
x = ones(2) # initial point
x16 = Float16.(x) # initial point in Float16
x32 = Float32.(x) # initial point in Float32
MPmodel = FPMPNLPModel(f,x32,FP); # will use interval arithmetic for error evaluation
f16, omega_f16 = objerrmp(MPmodel,x16) # evaluate objective and error bound at x with T[1] = Float16 FP model
f32, omega_f32 = objerrmp(MPmodel,x32) # evaluate objective and error bound at x with T[1] = Float16 FP model
g16, omega_g16 = graderrmp(MPmodel,x16) # evaluate gradient and error bound at x with T[2] = Float32 FP model
g32, omega_g32 = graderrmp(MPmodel,x32) # evaluate gradient and error bound at x with T[2] = Float32 FP model
```

**FPMPNLPModel Example 2: Relative error bounds**
```julia
using MultiPrecisionR2
using ADNLPModels

FP = [Float16, Float32] # selected FP formats
f(x) = x[1]^2 + x[2]^2 # objective function
x = ones(2) # initial point
x16 = Float16.(x) # initial point in Float16
x32 = Float32.(x) # initial point in Float32
ωfRelErr = [0.1,0.01] # objective error: 10% with Float16 and 1% with Float32
ωgRelErr = [0.05,0.02] # gradient error (norm): 5% with Float16 and 2% with Float32
MPmodel = FPMPNLPModel(f,x32,FP;ωfRelErr=ωfRelErr,ωgRelErr=ωgRelErr);
f32, omega_f32 = objerrmp(MPmodel,x32) # evaluate objective and error bound at x with T[1] = Float32 FP model
g16, omega_g16 = graderrmp(MPmodel,x16) # evaluate gradient and error bound at x with T[2] = Float16 FP model
```

**FPMPNLPModel Example 3: Mixed Interval/Relative Error Bounds**
```julia
using MultiPrecisionR2
using IntervalArithmetic
using ADNLPModels

setrounding(Interval,:accurate)
FP = [Float16, Float32] # selected FP formats
f(x) = x[1]^2 + x[2]^2 # objective function
x = ones(2) # initial point
x16 = Float16.(x) # initial point in Float16
x32 = Float32.(x) # initial point in Float32
ωgRelErr = [0.05,0.02] # gradient error (norm): 5% with Float16 and 2% with Float32
MPmodel = FPMPNLPModel(f,x32,FP; ωgRelErr = ωgRelErr) # use interval for objective error bound and relative error bound for gradient
f32, omega_f32 = objerrmp(MPmodel,x32) # evaluate objective and error bound with interval at x with T[1] = Float32 FP model
g16, omega_g16 = graderrmp(MPmodel,x16) # evaluate gradient and error bound at x with T[2] = Float16 FP model
```

**FPMPNLPModel Example 4: Interval evaluation is slow**
```julia
using MultiPrecisionR2
using IntervalArithmetic
using ADNLPModels

setrounding(Interval,:accurate)
FP = [Float32] # selected FP formats
n = 1000
f(x) = sum([x[i]^2 for i =1:n])  # objective function
x = ones(n) # initial point
x32 = Float32.(x) # initial point in Float32
MPmodelInterval = FPMPNLPModel(f,x32,FP) # use interval for objective and gradient error bounds
ωfRelErr = [Float64(sqrt(eps(Float32)))] # objective error
ωgRelErr = [Float64(sqrt(eps(Float32)))] # gradient error (norm)
MPmodelRelative = FPMPNLPModel(f,x32,FP,ωfRelErr = ωfRelErr, ωgRelErr = ωgRelErr) # will use relative error bounds
# precompile
objerrmp(MPmodelInterval,x32)
objerrmp(MPmodelRelative,x32)
graderrmp(MPmodelInterval,x32)
graderrmp(MPmodelRelative,x32)

@time objerrmp(MPmodelInterval,x32) # interval evaluation of objective
@time objerrmp(MPmodelRelative,x32) # classic evaluation of objective
@time graderrmp(MPmodelInterval,x32) # interval evaluation of gradient
@time graderrmp(MPmodelRelative,x32) # classic evaluation of gradient
```

### **High Precision Format**
`FPMPNLPModel` performs some operations with a high precision FP format to provide more numerical stability. The convergence of MPR2 relies on the fact that such operations are "exact", as if performed with infinite precision. This high precision format is also used by `MPR2solve` in a similar way and for the same reasons (see section [High Precision Format: MPR2 Solver](#high-precision-format-mpr2-solver))

This high precision format is `FPMPNLPModel.HPFormat` and can be given as a keyword argument upon instanciation. The default value is `Float64`. Note that this high precision format also correspond to the type parameter `H` in `struct FPMPNLPModel{H,F,T<:Tuple}`. It is expected that `FPMPNLPModel.HPFormat` has at least equal or greater machine epsilon than the highest precision FP format that can be used for objective or gradient evaluation.

**FPMPNLPModel Example 5: HPFormat value**

```julia
using MultiPrecisionR2
using IntervalArithmetic
using ADNLPModels

setrounding(Interval,:accurate)
FP = [Float16, Float32, Float64] # selected FP formats, max eval precision is Float64
f(x) = x[1]^2 + x[2]^2 # objective function
x = ones(2) # initial point
MPmodel = FPMPNLPModel(f,x,FP); # throws warning
MPmodel = FPMPNLPModel(f,x,FP,HPFormat = Float32); # throws error
```

### **Gradient and Dot Product Error**: Gamma Callback Function
`FPMPNLPModel.graderrmp()` computes both the gradient and the error bound $\omega_g$ as in the [Notations](#notations) section. To do so, it is necessary to compute norm of the gradient and therefore to take the related error into account, given by the $\beta$ function (see section [Rounding Errors Handling](#rounding-errors-handling)). Note that $\beta$ expresses with $\gamma$ function which models the dot product error. An implicit condition is that $\gamma(n,u_{max}) \leq 1$, with $u_{max}$ the smallest unit-roundoff among the provided FP formats.

For example, if the highest precision format Float32 is used, $u_{max} \approx 1 e^{-7}$ which limits the size of the problems that can be solve to $\approx 1e^{7}$ variables with the default implementation $γ(n,u) = n*u$. If this is a problem, the user can provide its own callback function `FPMPNLPModels.γfunc`. This is illustrated in the example below.

**FPMPNLPModel Example 5: Gradient and Dot Product Error**
The code below returns an error at the instanciation of `FPMPNLPModels` indicating that the dimension of the problem is too big with respect to the highest precision FP format provided (`Float16`).
```julia
using MultiPrecisionR2
using IntervalArithmetic
using ADNLPModels

setrounding(Interval,:accurate)
FP = [Float16] # limits the size of the problem to n = 1/eps(Float16) (= 1000)
dim = 2000 # dimension of the problem too large
f(x) =sum([x[i]^2 for i=1:dim]) # objective function
x = ones(Float16,dim) # initial point
MPmodel = FPMPNLPModel(f,x,FP); # throw error
```

The user can provide a less pressimistic $\gamma$ function for dot product error bound.
**Warning:** Providing your own $\gamma$ function can break the convergence properties of MPR2.
```julia
gamma(n,u) = sqrt(n)*u # user defined γ function, less pessimistic than n*u used by default
MPmodel = FPMPNLPModel(f,x,FP,γfunc = gamma); # no error since sqrt(dim)*eps(Float16) < 1
```

## **MPR2Solver**
See `MPR2Solver` documentation.

### **Gamma Function**
As mentionned in section [Rounding Errors Handling](#rounding-errors-handling), MPR2 relies on the dot product error function $\gamma$ to handle some rounding errors (norm and model decrease). The $\gamma$ function used by MPR2 is the one of the `solve!()` function's `FPMPNLPModel` argument. 

### **High Precision Format: MPR2 Solver**
MPR2 uses a high precision format to compute "exactly" the values that are *defined* (see section [MPR2 Algorithm Broad Description](#mpr2-algorithm-broad-description-differs-from-package-implementation)). This high precision format corresponds to the type parameter `H` in `MultiPrecisionR2.solve!()`. The high precision format used by MPR2 is `FPMPNLPModel.HPFormat` of `MPnlp` argument of `MultiPrecisionR2.solve!()` (see section [High Precision Format](#high-precision-format)).

### **Lack of Precision**
The default implementation of MPR2 stops when the condition on the objective evaluation error or the $\mu$ indicator fails with the highest precision evaluations (see Lines 7,8,10 of MPR2 algorithm in section [MPR2 Algorithm Broad Description](#mpr2-algorithm-broad-description-differs-from-package-implementation)). If this happens, MPR2 returns the ad-hoc warning message. The user can tune MPR2 parameters to try to avoid such early stop via the structure `MPR2Params` and provide it as a keyword argument to `solve!`. Typically, 
* **If the objective error is too big**: the user should increase $\eta_0$ parameter.
* **If $\mu_k$ is too big**: the user should increase $\kappa_m$ parameter.
The user has to make sure that the parameters respect the convergence conditions (see section [Conditions on Parameters](#conditions-on-parameters)).

**MPR2Solver Example 1**: Lack of Precision and Parameters Selection
```julia
using MultiPrecisionR2
using IntervalArithmetic
using ADNLPModels

setrounding(Interval,:accurate)
FP = [Float16, Float32] # selected FP formats, max eval precision is Float64
f(x) = (1-x[1])^2 + 100*(x[2]-x[1]^2)^2 # Rosenbrock function
x = Float32.(1.5*ones(2)) # initial point
HPFormat = Float64
MPmodel = FPMPNLPModel(f,x,FP,HPFormat = HPFormat);
solver = MPR2Solver(MPmodel);
stat = MPR2(MPmodel) 
```
Running the above code block returns a warning indicating that R2 stops because the error on the objective function is too big to ensure convergence. The problem can be overcome in this example by tolerating more error on the objective by increasing $\eta_0$.

```julia
η₀ = 0.1 # greater than default value 0.01
η₁ = 0.3
η₂ = 0.7
κₘ = 0.1
γ₁ = Float16(1/2) # must be FP format of lowest evaluation precision for numerical stability
γ₂ = Float16(2) # must be FP format of lowest evaluation precision for numerical stability
param = MPR2Params(η₀,η₁,η₂,κₘ,γ₁,γ₂)
stat = MPR2(MPmodel,par = param) 
```
Now MPR2 converges to a first order critical point since we tolerate enough error on the objective evaluation.

### **Evaluation Counters**

MPR2 counts the number of objective and gradient evaluations are counted for each FP formats. They are stored in `counters` field of the `FPNLPModel` structure. The `counters` field is a `MPCounters`.

```julia
setrounding(Interval,:accurate)
FP = [Float16, Float32, Float64]
f(x) = sum(x.^2) 
x0 = ones(10)
HPFormat = Float64
MPmodel = FPMPNLPModel(f,x0,FP,HPFormat = HPFormat);
MPR2(MPmodel,verbose=1)
@show MPmodel.counters.neval_obj # numbers of objective evaluations 
@show MPmodel.counters.neval_grad # numbers of gradient evaluations
```

# Advanced Use

`MultiPrecisionR2.jl` does more than implementing MPR2 algorithm as describes in Section [MPR2 Algorithm General Description](#mpr2-algorithm-general-description). `MPR2Precision.jl` enables the user to define its own strategy to select evaluation precisions and to handle evaluation errors. This is made possible by using callback functions when calling `MultiPrecisionR2.solve!()`. The default implementation of `MultiPrecisionR2.solve!()` relies on specific implementation of these callback functions, which are included in the package. The user is free to provide its own callback functions to change the behavior of the algorithm. 

## Diving into MPR2 Implementation
`MPR2Precision.jl` relies on callback functions that handle the objective and gradient evaluation. These callback functions are expected to compute values for the objective and the gradient and handle the evaluation precision. In the default implementation, the conditions on the error bound of objective and gradient evaluation are dealt with in these callback functions. That is why such convergence conditions (which user might choose not to implement) does not appear in the minimal implementation description of the code below.

### **Minimal Implementation Description**

Here is a minimal description of the implementation of `MultiPrecisionR2.solve!()` to understand when and why these functions are called. Please refer to the implementation for details. Note that the callback functions templates are not respected for the sake of explanation.
```julia
solve!(solver::MPR2solver{T},MPnlp::FPMPNLPModel{H};kwargs...)
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
    if # not enough precision with max precision
      break
    end
    if # gradient recomputed by recompute_g!() 
      # recompute step with new value of the gradient
      # check overflow
      # recompute candidate with new step
      # check overflow
      # recompute model decrease with new gradient and step
    end
    compute_f_at_x() # possibly recompute f(x) and ωf(x) if ωf(x) is too big
    if # not enough precision with max precision (ωf(x) too big)
      break
    end
    compute_f_at_c!() # compute f(c) and ωf(c)
    if # not enough precision with max precision (ωf(c) too big)
      break
    end
    # ...
    # compute ρ = (f(x) - f(c))/ΔT
    # update x and σ
    # ... 
    if # ρ ≥ η₁
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