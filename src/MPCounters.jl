export MPCounters, increment!, reset!

"""
    MPCounters

Struct for storing the number of function evaluations with each FP formats.

---

    MPCounters()

Creates an empty Counters struct.
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

  function MPCounters()
    return new([Dict{DataType,Int}() for _ in fieldnames(MPCounters)]...)
  end
end

# simple default API for retrieving counters
for mpcounter in fieldnames(MPCounters)
  @eval begin
    """
        $($mpcounter)(nlp)

    Get the number of `$(split("$($mpcounter)", "_")[2])` evaluations.
    """
    $mpcounter(nlp::AbstractMPNLPModel) = nlp.counters.$mpcounter
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
      if haskey(nlp.counters.$fun,T)
        nlp.counters.$fun[T] += 1
      else
        push!(nlp.counters.$fun,T=>1)
      end
    end
  end
end

# """
#     decrement!(nlp, s)

# Decrement counter `s` of problem `nlp`.
# """
# function decrement!(nlp::AbstractNLPModel, s::Symbol)
#   setproperty!(nlp.counters, s, getproperty(nlp.counters, s) - 1)
# end

# """
#     sum_counters(counters)

# Sum all counters of `counters` except `cons`, `jac`, `jprod` and `jtprod`.
# """
# function sum_counters(c::Counters)
#   sum = 0
#   for x in fieldnames(Counters)
#     if !(x in (:neval_cons, :neval_jac, :neval_jprod, :neval_jtprod))
#       sum += getproperty(c, x)
#     end
#   end
#   return sum
# end
# """
#     sum_counters(nlp)

# Sum all counters of problem `nlp` except `cons`, `jac`, `jprod` and `jtprod`.
# """
# sum_counters(nlp::AbstractNLPModel) = sum_counters(nlp.counters)

"""
    reset!(counters)

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
    reset!(nlp)

Reset evaluation count and model data (if appropriate) in `nlp`.
"""
function reset!(nlp::AbstractMPNLPModel)
  reset!(nlp.counters)
  reset_data!(nlp.Model)
  return nlp
end