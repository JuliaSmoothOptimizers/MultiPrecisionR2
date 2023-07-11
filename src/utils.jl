"""
    ObjIntervalEval_test(nlp::AbstractNLPModel,FPList::AbstractArray)

Test interval evaluation of objective for all formats in `FPList`.
Test fails and return an error if:
  * Interval evaluation returns an error
  * Interval evaluation is not type stable
See [`FPMPNLPModel`](@ref), [`AbstractNLPModel`](@ref)
"""
function ObjIntervalEval_test(nlp::AbstractNLPModel,FPList::AbstractArray)
  for fp in FPList
    @debug "Testing objective interval evaluation with $fp "
    try
      X0 = [fp(xi)..fp(xi) for xi ∈ nlp.meta.x0]
      intype = fp
      output = obj(nlp,X0) # ! obj(nlp,X0::IntervalBox{T}) returns either ::T or Interval{T}
      outtype = typeof(output) <: AbstractFloat ? typeof(output) : typeof(output.lo) 
      if intype != outtype
        @error "Interval evaluation of objective function not type stable ($intype -> $outtype)"
        error("Interval evaluation of objective function not type stable ($intype -> $outtype)")
      end
    catch e
      error("Objective function evaluation error with interval, error model must be provided.\n 
      Error detail:")
      @show e
    end
  end
end

"""
    GradIntervalEval_test(nlp::AbstractNLPModel,FPList::AbstractArray)

Test interval evaluation of gradient for all FP formats.
Test fails and return an error if:
  * Interval evaluation returns an error
  * Interval evaluation is not type stable
See [`FPMPNLPModel`](@ref), [`AbstractNLPModel`](@ref)
"""
function GradIntervalEval_test(nlp::AbstractNLPModel,FPList::AbstractArray)
  for fp in FPList
    @debug "Testing grad interval evaluation with $fp"
    try 
      X0 = [fp(xi)..fp(xi) for xi ∈ nlp.meta.x0]
      intype = fp
      output = grad(nlp,X0)
      outtype = typeof(output[1]) <: AbstractFloat ? typeof(output[1]) : typeof(output[1].lo) 
      if intype != outtype
        @error "Interval evaluation of gradient not type stable ($intype -> $outtype)"
        error("Interval evaluation of gradient not type stable ($intype -> $outtype)")
      end
    catch e
      error("Gradient evaluation error with interval, error model must be provided.\n 
      Error detail:")
      @show e
    end
  end
end

"""
    γfunc_test_template(γfunc)

Tests if γfunc callback function is properly implemented.
Expected template: γfunc(n::Int,u::Float) -> Float
"""    
function γfunc_test_template(γfunc)
  err_msg = "Wrong γfunc template, expected template: γfunc(n::Int,u::Float) -> Float"
  try
    typeof(γfunc(1,1.0)) <: AbstractFloat  || error(err_msg)
  catch e
    error(err_msg)
  end
end

"""
    γfunc_test_error_bound(n::Int,eps::AbstractFloat,γfunc)

Tests if γfunc callback provides strictly less than 100% error for dot product error of vector
of size the dimension of the problem and the lowest machine epsilon.
"""
function γfunc_test_error_bound(n::Int,eps::AbstractFloat,γfunc)
  err_msg = "γfunc: dot product error greater than 100% with highest precision. Consider using higher precision floating point format, or provide a different callback function for γfunc (last option might cause numerical instability)."
  if γfunc(n,eps) >= 1.0
    error(err_msg)
  end
end

"""
    check_overflow(f)

* f::AbstractFloat: Returns true if `f` is inf or nan, false otherwise.
* f::Interval : Returns true if `diam(f)` is inf or nan, false otherwise.
* f::AbstractVector{AbstractFloat} : Returns true if on element of `f` is inf or nan, false otherwise.
* f::AbstractVector{Interval} : Returns true if on element of `diam(f)` is inf, false otherwise.
"""
function check_overflow(f::AbstractFloat)
  return isinf(f) || isnan(f)
end

function check_overflow(f::Interval)
  return isinf(diam(f))
end

function check_overflow(g::AbstractVector{T}) where {T <: Interval}
  return findfirst(x->isinf(diam(x)),g) !== nothing
end

function check_overflow(g::AbstractVector{T}) where {T <: AbstractFloat}
  return findfirst(x->isinf(x) || isnan(x),g) !== nothing # one element of g overflow ? true : false
end