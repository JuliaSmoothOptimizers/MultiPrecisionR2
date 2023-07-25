# Disclaimer

The reader is encouraged to read the `FPMPNLPModel` tutorial before this tutorial.

# Motivation
Here is the comparison between `R2` algorithm from `JSOSolvers.jl` run with `Float64` and `MPR2` from `MultiPrecisionR2.jl` run with `Float16`, `Float32` and `Float64` over a set of unconstrained problems taken from `OptimizationProblems.jl`.
The time and energy savings offered by MPR2 compared with R2 is estimated with the rule of thumb:

Number of bits divided by two $\implies$ time computation divided by two and energy consumption divided by four.


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

# MPR2 Algorithm: General Description and Basic Use

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

In the algorithm, *compute* means compute with finite-precision machine computation, and *define* means "compute exactly", *i.e.* with infinite precision. Defining a value is therefore not possible to perform on a (finite-precision) machine. This point is discussed later.

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
  where $x$ and $y$ are two FP vectors (same FP format) of dimension $n$ and $u$ is the round-off unit of the FP format. This is a crude yet guaranteed upper bound on the error. The $\gamma$ function is embedded in the `FPMPNLPModel` structure as a callback (see documentation and tutorial), such that the user can choose its own formula.
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

MPR2 solver relies on multi-precision models structure `FPMPNLPModels` (Floating Point Multi Precision Non Linear Programming Models), that derives from [`NLPModels.jl`](https://github.com/JuliaSmoothOptimizers/NLPModels.jl). This structure embeds the problem and provides the interfaces to evaluate the objective and the gradient with several FP formats. These evaluation functions are used by `MPR2Solver` to evaluate the objective and gradient with different FP format. For details about `FPMPNLPModels`, see related documentaion.

## **MPR2Solver**
See `MPR2Solver` documentation.

### **Gamma Function**
MPR2 relies on the dot product error function $\gamma$ to handle some rounding errors (norm and model decrease). The $\gamma$ function used by MPR2 is the one of the `MPR2()` or `solve!()` `FPMPNLPModel` argument. 

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