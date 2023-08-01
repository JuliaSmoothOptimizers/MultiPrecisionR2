# Evaluation Mode

`FPMPNLPModel` implements tools to evaluate the objective and the gradient and the error bounds $\omega_f(x_k)$ and $\omega_g(x_k)$ with two modes:
1. **Relative Error**: The error bounds $\omega_f(x_k)$ and $\omega_g(x_k)$ are **estimated** as fractions of $|\hat{f}(x_k)|$ and $\|\hat{g}(x_k)\|$, respectively. It is to the user to provide these fraction values.  
By default, interval arithmetic is used for error bound computation. If relative error bounds have to be used, `ωfRelErr` and `ωgRelErr` keyword arguments must be used when calling the constructor function.
2. **Interval Arithmetic**: Using interval arithmetic enables to track rounding errors and to provide guaranteed bounds.`MultiPrecisionR2.jl` relies on [`IntervalArithmetic.jl`](https://juliaintervals.github.io/pages/packages/intervalarithmetic/) library to perform interval evaluations of the gradient and the objective.  
  **Warning**: Using interval evaluation can lead to significantly long computation time. Interval evaluation might fail for some objective functions due to `IntervalArithmetic.jl` errors.

**FPMPNLPModel Example 1: Interval Arithmetic for error bounds**

```@example
using MultiPrecisionR2
using IntervalArithmetic

setrounding(Interval,:accurate) # add this to avoid error with Float16 interval evaluations
FP = [Float16, Float32] # selected FP formats
f(x) = x[1]^2 + x[2]^2 # objective function
x = ones(2) # initial point
x16 = Float16.(x) # initial point in Float16
x32 = Float32.(x) # initial point in Float32
MPmodel = FPMPNLPModel(f,x32,FP, obj_int_eval = true, grad_int_eval = true); # will use interval arithmetic for error evaluation
f16, omega_f16 = objerrmp(MPmodel,x16) # evaluate objective and error bound at x with T[1] = Float16 FP model
f32, omega_f32 = objerrmp(MPmodel,x32) # evaluate objective and error bound at x with T[1] = Float16 FP model
g16, omega_g16 = graderrmp(MPmodel,x16) # evaluate gradient and error bound at x with T[2] = Float32 FP model
g32, omega_g32 = graderrmp(MPmodel,x32) # evaluate gradient and error bound at x with T[2] = Float32 FP model
```

**FPMPNLPModel Example 2: Relative error bounds**

```@example
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

It is possible to evaluate the objective with interval mode and the gradient with relative error mode, and vice-versa.

```@example
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
MPmodel = FPMPNLPModel(f,x32,FP; obj_int_err = true, ωgRelErr = ωgRelErr) # use interval for objective error bound and relative error bound for gradient
f32, omega_f32 = objerrmp(MPmodel,x32) # evaluate objective and error bound with interval at x with T[1] = Float32 FP model
g16, omega_g16 = graderrmp(MPmodel,x16) # evaluate gradient and error bound at x with T[2] = Float16 FP model
```

**FPMPNLPModel Example 4: Interval evaluation is slow**

Interval evaluation provides guaranteed bounds, but is slow compared with classical evaluation.

```@example
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
MPmodelRelative = FPMPNLPModel(f,x32,FP) # will use default relative error bounds
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

# **High Precision Format**
`FPMPNLPModel` performs some operations with a high precision FP format to provide more numerical stability. The convergence of MPR2 relies on the fact that such operations are "exact", as if performed with infinite precision.

This high precision format can be given as a keyword argument upon instantiation of `FPMPNLPModel`. The default value is `Float64`. Note that this high precision format corresponds to the type parameter `H` in `struct FPMPNLPModel{H,F,T<:Tuple}`. It is expected that `FPMPNLPModel.HPFormat` has at least equal or greater machine epsilon than the highest precision FP format that can be used for objective or gradient evaluation.

**FPMPNLPModel Example 5: HPFormat value**

```@example
using MultiPrecisionR2

FP = [Float16, Float32, Float64] # selected FP formats, max eval precision is Float64
f(x) = x[1]^2 + x[2]^2 # objective function
x = ones(2) # initial point
MPmodel = FPMPNLPModel(f,x,FP); # throws warning
try
  MPmodel = FPMPNLPModel(f,x,FP,HPFormat = Float32); # throws error
catch e
  e
end
```

# **Gradient and Dot Product Error**: Gamma Callback Function

`FPMPNLPModel.graderrmp()` computes both the gradient and the error relative error bound $\omega_g$ such that = $||\nabla f(x) - fl(\nabla f(x))||_2 \leq \omega_g ||fl(\nabla f(x))||_2$. To do so, it is necessary to compute norm of the gradient and therefore to take the related error into account, given by the $\beta$ function:
$\beta(n,u) = \max(|\sqrt{\gamma(n,u)-1}-1|,|\sqrt{\gamma(n,u)+1}-1|)$
which expresses with $\gamma$ function which models the dot product error. This $\gamma$ function is a callback that can be provided to the `FPMPNLPModel` with the `γfunc`, and by default is $γfunc(n,u) = n*u$. An implicit condition is that $\gamma(n,u_{max}) \leq 1$, with $u_{max}$ the smallest unit-roundoff among the FP formats in `FPList`.

For example, if the highest precision format `Float32` is used, $u_{max} \approx 1 e^{-7}$ which limits the size of the problems that can be solved to $\approx 1e^{7}$ variables with the default implementation $γ(n,u) = n*u$. If this is a problem, the user can provide its own callback function `FPMPNLPModels.γfunc`. This is illustrated in the example below.

**FPMPNLPModel Example 5: Gradient and Dot Product Error**

The code below returns an error at the instantiation of `FPMPNLPModels` indicating that the dimension of the problem is too big with respect to the highest precision FP format provided (`Float16`).
```@example ex
using MultiPrecisionR2

FP = [Float16] # limits the size of the problem to n = 1/eps(Float16) (= 1000)
dim = 2000 # dimension of the problem too large
f(x) =sum([x[i]^2 for i=1:dim]) # objective function
x = ones(Float16,dim) # initial point
try
  MPmodel = FPMPNLPModel(f,x,FP); # throw error
catch e
  e
end
```

The user can provide a less pessimistic $\gamma$ function for dot product error bound.
**Warning:** Providing your own $\gamma$ function might increase numerical instability of MPR2.
```@example ex
using MultiPrecisionR2

gamma(n,u) = sqrt(n)*u # user defined γ function, less pessimistic than n*u used by default
MPmodel = FPMPNLPModel(f,x,FP,γfunc = gamma); # no error since sqrt(dim)*eps(Float16) < 1
```
