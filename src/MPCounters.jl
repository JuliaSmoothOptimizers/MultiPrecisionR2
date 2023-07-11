export MPCounters, increment!, reset!

"""
    MPCounters

Struct for storing the number of function evaluations with each floating point format.
The fields are the same as [NLPModels.Counters](https://jso.dev/NLPModels.jl/stable/reference/#NLPModels.Counters),
but contains a `Dict{DataType,Int}`.

---

    MPCounters(FPformats::Vector{DataType})

Creates an empty MPCounters struct for types in the vector `FPformats`.

```julia
using MultiPrecisionR2.jl
FPformats = [Float16, Float32]
cntrs = MPCounters(FPformats)
```
"""
mutable struct MPCounters
  neval_obj::Dict{DataType,Int}  # Number of objective evaluations.
  neval_grad::Dict{DataType,Int}  # Number of objective gradient evaluations.
  neval_cons::Dict{DataType,Int}  # Number of constraint vector evaluations.
  neval_cons_lin::Dict{DataType,Int}  # Number of linear constraint vector evaluations.
  neval_cons_nln::Dict{DataType,Int}  # Number of nonlinear constraint vector evaluations.
  neval_jcon::Dict{DataType,Int}  # Number of individual constraint evaluations.
  neval_jgrad::Dict{DataType,Int}  # Number of individual constraint gradient evaluations.
  neval_jac::Dict{DataType,Int}  # Number of constraint Jacobian evaluations.
  neval_jac_lin::Dict{DataType,Int}  # Number of linear constraints Jacobian evaluations.
  neval_jac_nln::Dict{DataType,Int}  # Number of nonlinear constraints Jacobian evaluations.
  neval_jprod::Dict{DataType,Int}  # Number of Jacobian-vector products.
  neval_jprod_lin::Dict{DataType,Int}  # Number of linear constraints Jacobian-vector products.
  neval_jprod_nln::Dict{DataType,Int}  # Number of nonlinear constraints Jacobian-vector products.
  neval_jtprod::Dict{DataType,Int}  # Number of transposed Jacobian-vector products.
  neval_jtprod_lin::Dict{DataType,Int}  # Number of transposed linear constraints Jacobian-vector products.
  neval_jtprod_nln::Dict{DataType,Int}  # Number of transposed nonlinear constraints Jacobian-vector products.
  neval_hess::Dict{DataType,Int}  # Number of Lagrangian/objective Hessian evaluations.
  neval_hprod::Dict{DataType,Int}  # Number of Lagrangian/objective Hessian-vector products.
  neval_jhess::Dict{DataType,Int}  # Number of individual Lagrangian Hessian evaluations.
  neval_jhprod::Dict{DataType,Int}  # Number of individual constraint Hessian-vector products.

  function MPCounters(FPformats::Vector{DataType})
    return new([Dict([f => 0 for f in FPformats]) for _ in fieldnames(MPCounters)]...)
  end
end

# simple default API for retrieving counters
for mpcounter in fieldnames(MPCounters)
  @eval begin
    """
        $($mpcounter)(nlp)
        $($mpcounter)(nlp,T)

    Get the total number (all FP formats) of `$(split("$($mpcounter)", "_")[2])` evaluations.
    If extra argument T is provided, returns  turn number of `$(split("$($mpcounter)", "_")[2])` evaluations for the given FP format T.
    """
    NLPModels.$mpcounter(nlp::AbstractMPNLPModel) = sum(collect(values(nlp.counters.$mpcounter)))
    NLPModels.$mpcounter(nlp::AbstractMPNLPModel,T::DataType) = nlp.counters.$mpcounter[T]
    export $mpcounter
  end
end

"""
    increment!(nlp, s)

Increment counter `s` of problem `nlp`.
"""
@inline function increment!(nlp::AbstractMPNLPModel, s::Symbol, T::DataType)
  increment!(nlp, Val(s), T)
end

for fun in fieldnames(MPCounters)
  @eval begin 
    function increment!(nlp::AbstractMPNLPModel, ::Val{$(Meta.quot(fun))}, T::DataType)
      nlp.counters.$fun[T] += 1
    end
  end
end

"""
    decrement!(mpnlp, s, FPFormat)

Decrement counter `s` of problem `mpnlp` for the given FPFormat provided.
"""

function decrement!(mpnlp::AbstractNLPModel, s::Symbol, FPFormat::DataType)
  counter = getproperty(mpnlp.counters, s)
  counter[FPFormat] -= 1
  setproperty!(mpnlp.counters, s, counter)
end

"""
    sum_counters(c::MPCounters)

Sum all counters of `counters` except `cons`, `jac`, `jprod` and `jtprod`.
"""
function sum_counters(c::MPCounters)
  sum = Dict{DataType,Int}()
  for x in fieldnames(MPCounters)
    if !(x in (:neval_cons, :neval_jac, :neval_jprod, :neval_jtprod))
      mergewith(+,sum,getproperty(c, x))
    end
  end
  return sum
end
"""
    sum_counters(mpnlp)

Sum all counters of problem `nlp` except `cons`, `jac`, `jprod` and `jtprod`.
"""
sum_counters(mpnlp::AbstractMPNLPModel) = sum_counters(mpnlp.counters)

"""
    reset!(counters::MPCounters)

Reset evaluation counters
"""
function reset!(counters::MPCounters)
  for f in fieldnames(MPCounters)
    for key in keys(f)
      f[key] = 0
    end
  end
  return counters
end

"""
    reset!(mpnlp::AbstractMPNLPModel)

Reset evaluation count and model data (if appropriate) in `mpnlp`.
"""
function reset!(mpnlp::AbstractMPNLPModel)
  reset!(mpnlp.counters)
  reset_data!(mpnlp.Model)
  return mpnlp
end
