using LinearAlgebra

import NLPModels: obj, grad, grad!, objgrad!, objgrad

export AbstractMPNLPModel, FPMPNLPModel, obj, grad, grad!, objerrmp, graderrmp, graderrmp!

const INT_ERR = 0
const REL_ERR = 1

"""
    FPMPNLPModel(Model::AbstractNLPModel{D,S},FPList::Vector{K}; kwargs...) where {D,S,K<:DataType}

Floating-Point Multi-Precision Non Linear Model structure. This structure is intended to be used as MPR2Solver input.

Primairly stores NLPmodels instanciated with different Floating Point formats and provide errors on objective function and grandient evaluation (see `objerrmp` and `graderrmp`).
The error models are :
+ ojective: |fl(f(x)) - f(x)| ≤ ωf
+ gradient: |fl(∇f(x)) - ∇f(x)| ≤ ||fl(∇f(x))||₂ ωg.
ωf and ωg are needed for MPR2Solver. They are evaluated either using:
+ interval analysis (can be very slow)
+ based on relative error assumption (see `ωfRelErr` and `ωgRelErr` field description below)   


# Fields
- `Model::AbstractNLPModel` : NLPModel
- `FPList::Vector{DataType}` : List of floating point formats
- `EpsList::Vector{H}` : List of machine epsilons of the floating point formats
- `HPFormat::DataType` : High precision floating point format, used for intermediate value computation in MPR2Solver. Is H parameter.
- `γfunc` : callback function for dot product rounding error parameter |γ|, |fl(x.y) - x.y| ≤ |x|.|y| γ. Expected signature is `γfunc(n::Int,u::H)` and output is `H`. Default callback `γfunc(n::Int,u::H) = n*u` is implemented upon instanciation. 
- `ωfRelErr::Vector{H}` : List of relative error factor for objective function evaluation for formats in `FPList`. Error model is |f(x)-fl(f(x))| ≤ ωfRelErr * |fl(f(x))| 
- `ωgRelErr::Vector{H}` : List of relative error factor for gradient evaluation for formats in `FPList`. Error model is |∇f(x)-fl(∇f(x))| ≤ ωgRelErr * ||fl(∇f(x))||₂ 
- `ObjEvalMode::Int` : Evalutation mode for objective and error. Set automatically upon instanciation. Possible values:
  + `INT_ERR` : interval evaluation of objective (chosen as middle of the interval) and error
  + `REL_ERR` : classical evaluation and use relative error model (with ωfRelErr value)
- `GradEvalMode::Int` : Evalutation mode for gradient and error. Set automatically upon instanciation. Possible values:
  + `INT_ERR` : interval evaluation of gradient (chosen as middle of interval vector) and error
  + `REL_ERR` : classical evaluation and use relative error model (with ωgRelErr value)

# Constructors:
- `FPMPModel(Model::AbstractNLPModel, FPList::Vector{DataType}; nvar=100, kwargs...)` : create a FPMPModel from Model with FPList precisions
- `FPMPModel(s::symbol,FPList::Vector{DataType}; nvar=100, kwargs...)` : create a FPMPModel from a symbol linked to an AbstractNLPModel.
- `FPModels`(f,x0::Vector,FPList::Vector{DataType}; nvar=100, kwargs...)

Keyword arguments: 
- nvar: dimension of the problem (if scalable)
- kwargs: 
  + `HPFormat=Float64` : high precision format (must be at least as accurate as FPList[end])
  + `γfunc=nothing` : use default if not provided (see Fields section above)
  + `ωfRelErr=nothing` : use interval evaluation if not provided
  + `ωgRelErr=nothing` : use interval evaluation if not provided

# Checks upon instanciation
Some checks are performed upon instanciation. These checks include:
+ Length consistency of vector fields:  FPList, EpsList
+ HPFormat is at least as accurate as the highest precision floating point format in `FPList``. Ideally HPFormat is more accurate to ensure the numerical stability of MPR2 algorithm.
+ Interval evaluations: it might happen that interval evaluation of objective function and/or gradient is type-unstable or returns an error. The constructor returns an error in this case. This type of error is most likely due to `IntervalArithmetic.jl`.
+ FPList is ordered by increasing floating point format accuracy

This checks can return `@warn` or `error`. 

# Examples 
```julia
T = [Float16, Float32]
f(x) = x[1]^2 + x[2]^2
x = zeros(2)
MPmodel = FPMPNLPModel(f,x0,T)
```

```julia
T = [Float16, Float32]
problems = setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
s = problem[end]
MPmodel = FPMPNLPModel(s,T)
```
"""
struct FPMPNLPModel{H,T<:Tuple,D,S} <: AbstractMPNLPModel{D,S}
  Model::AbstractNLPModel{D,S}
  meta::NLPModelMeta
  counters::MPCounters
  FPList::Vector{DataType}
  EpsList::Vector{H}
  UList::Vector{H}
  OFList::Vector{H}
  γfunc
  ωfRelErr::Vector{H}
  ωgRelErr::Vector{H}
  ObjEvalMode::Int
  GradEvalMode::Int
  X::T
  G::T
end

function FPMPNLPModel(Model::AbstractNLPModel{D,S},FPList::Vector{K};
  HPFormat=Float64,
  γfunc=nothing,
  ωfRelErr=nothing,
  ωgRelErr=nothing
) where {D,S,K<:DataType}
  EpsList = convert.(HPFormat,eps.(FPList))
  UList = EpsList.*HPFormat(1/2) # assume rounding mode is rounding to the nearest
  OFList = HPFormat.(prevfloat.(typemax.(FPList)))
  if sort(EpsList) != reverse(EpsList)
    err_msg = "FList must be ordered by increasing precision order (e.g [Float16,Float32,Float64])"
    @error err_msg
    error(err_msg)
  end
  if γfunc === nothing 
    γfunc = (n::Int,u::AbstractFloat) -> n*u
  else
    γfunc_test_template(γfunc) # test provided callback function
  end
  γfunc_test_error_bound(Model.meta.nvar,EpsList[end],γfunc)

  ObjEvalMode = INT_ERR
  if ωfRelErr === nothing
    @info "Interval evaluation used by default for objective error evaluation: might significantly increase computation time"
    ObjIntervalEval_test(Model,FPList)
    ωfRelErr=Vector{HPFormat}()
  else
    @lencheck length(FPList) ωfRelErr
    ObjEvalMode = REL_ERR
  end
  GradEvalMode = INT_ERR
  if ωgRelErr === nothing
    @info "Interval evaluation used by default for gradient error evaluation: might significantly increase computation time"
    GradIntervalEval_test(Model,FPList)
    ωgRelErr=Vector{HPFormat}()
  else
    @lencheck length(FPList) ωgRelErr
    GradEvalMode = REL_ERR
  end
  # instanciate interval containers X and G for point x and gradient g only if interval evaluation is used
  X = tuple()
  G = tuple()
  if ObjEvalMode == INT_ERR || GradEvalMode == INT_ERR
    X = Tuple([ElType(0) .. ElType(0) for _ in 1:Model.meta.nvar] for ElType in FPList)
    G = Tuple([ElType(0) .. ElType(0) for _ in 1:Model.meta.nvar] for ElType in FPList)
  end
  #reset counters : does not count interval evaluation of obj and grad for error test
  NLPModels.reset!(Model)
  EpsList[end] >= eps(HPFormat) || error("HPFormat ($HPFormat) must be a FP format with precision equal or greater than NLPModels (max prec NLPModel: $(FPList[end]))")
  EpsList[end] != eps(HPFormat) || @warn "HPFormat ($HPFormat) is the same format than highest accuracy NLPModel: chances of numerical instability increased"
  FPMPNLPModel(Model,Model.meta,MPCounters(FPList),FPList,EpsList,UList,OFList,γfunc,ωfRelErr,ωgRelErr,ObjEvalMode,GradEvalMode,X,G)
end

function FPMPNLPModel(s::Symbol,
  FPList::Vector{DataType};
  nvar::Int=100,
  kwargs...
)
  Model = eval(s)(type = Val(FPList[end]),n=nvar,gradient_backend = ADNLPModels.GenericForwardDiffADGradient)
  FPMPNLPModel(Model,kwargs...)
end

function FPMPNLPModel(f,x0,
  FPList::Vector{DataType};
  nvar::Int=100,
  kwargs...
)
  type = eltype(x0)
  if !(type in FPList)
    error("eltype of x0 ($type) must be in FPList ($FPList)")
  end
  Model = ADNLPModel(f,x0,n=nvar,gradient_backend = ADNLPModels.GenericForwardDiffADGradient)
  FPMPNLPModel(Model,FPList;kwargs...)
end

function Base.show(io::IO, m::FPMPNLPModel)
  print(io,m.Model)
end

function NLPModels.obj(m::FPMPNLPModel,x::AbstractVector)
  et = eltype(x)
  et<:Interval ? increment!(m,:neval_obj,typeof(x[1].lo)) : increment!(m,:neval_obj,eltype(x))
  obj(m.Model,x)
end

function grad(m::FPMPNLPModel,x::AbstractVector)
  et = eltype(x)
  et<:Interval ? increment!(m,:neval_obj,typeof(x[1].lo)) : increment!(m,:neval_obj,eltype(x))
  grad(m.Model,x)
end

function NLPModels.grad!(m::FPMPNLPModel,x::S,g::S) where S<:AbstractVector
  et = eltype(x)
  et<:Interval ? increment!(m,:neval_obj,typeof(x[1].lo)) : increment!(m,:neval_obj,eltype(x))
  grad!(m.Model,x,g)
end

"""
    objerrmp(m::FPMPNLPModel, x::V) where {S,V<:AbstractVector{S}}

Evaluate the objective and the evaluation error of the id-th model of m.
Inputs: x::Vector{S}
Outputs: ̂f::S, ωf <: AbstractFloat, with |f(x)-̂f| ≤ ωf
Overflow cases:
* Interval evaluation: overflow occurs if the diameter of the interval enclosing f(x) is Inf. Returns 0, Inf
* Classical evaluation:
  + If obj(x) = Inf: returns: Inf, Inf
  + If obj(x) != Inf and ωf = Inf, returns: obj(x), Inf  
"""
function objerrmp(m::FPMPNLPModel, x::V) where {S,V<:AbstractVector{S}}
  findfirst(t->t==S,m.FPList) !== nothing || error("Floating point format of x ($S) not supported by the multiprecison model (FP formats supported: $(m.FPList))")
  objerrmp(m, x, Val(m.ObjEvalMode))
end

@doc (@doc objerrmp)
function objerrmp(m::FPMPNLPModel, x::V, ::Val{INT_ERR}) where {S,V<:AbstractVector{S}}
  id = findfirst(t->t==S,m.FPList)
  for i in eachindex(x) # this is the proper way to instanciate interval vector, see issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546
    m.X[id][i] = x[i] .. x[i] # ::Vector{Interval{S}}
  end
  F = obj(m,m.X[id]) # ::Interval{S}
  if isinf(diam(F)) #overflow case
    return S(0.0), Inf
  else
    return mid(F), radius(F) #::S, ::S
  end
end

@doc (@doc objerrmp)
function objerrmp(m::FPMPNLPModel{H}, x::V, ::Val{REL_ERR}) where {H, S, V<:AbstractVector{S}}
  f = obj(m,x) #:: S
  if isinf(f) || isnan(f) #overflow case
    return f, Inf
  else
    id = findfirst(t->t==S,m.FPList)
    ωf = H(abs(f))*m.ωfRelErr[id] # Computed with H ≈> exact evaluation 
    return f, ωf # FP format of second returned value is H 
  end
end

"""
    graderrmp!(m::FPMPNLPModel{H}, x::V, g::V) where {H, S, V<:AbstractVector{S}}

Evaluate the gradient g and the relative evaluation error ωg of the id-th model of m.
Inputs: x::Vector{S} with S in m.FPList
Outputs: g::Vector{S}, ωg <: AbstractFloat satisfying: ||∇f(x) - g||₂ ≤ ωg||g||₂
Note: ωg FP format may be different than S
Overflow cases:
* Interval evaluation: if at least one element of g has infinite diameter, returns [0]ⁿ, Inf
* Classical evaluation: if one element of g overflow, returns g, Inf 
"""
function graderrmp!(m::FPMPNLPModel{H}, x::V, g::V) where {H, S, V<:AbstractVector{S}}
  findfirst(t->t==S,m.FPList) !== nothing || error("Floating point format of x ($S) not supported by the multiprecison model (FP formats supported: $(m.FPList))")
  graderrmp!(m, x, g, Val(m.GradEvalMode))
end

@doc( @doc graderrmp!)
function graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, ::Val{INT_ERR}) where {H, S, V<:AbstractVector{S}}
  id = findfirst(t->t==S,m.FPList)
  for i in eachindex(x) # this is the proper way to instanciate interval vector, see issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546
    m.X[id][i] = x[i] .. x[i] # ::Vector{Interval{S}}
  end
  grad!(m,m.X[id],m.G[id]) # ::IntervalBox{S}
  if findfirst(x->isinf(diam(x)),m.G[id]) !== nothing  #overflow case
    g .= zero(S)
    return Inf
  end
  g .= mid.(m.G[id]) # ::Vector{S}
  if findfirst(x->x!==S(0),g) === nothing # g = mid(G) == 0ⁿ
    if findfirst(x->radius(x)!==S(0),m.G[id]) === nothing # G = [0,0]ⁿ
      return 0
    else # G = [-Gᵢ,Gᵢ], pick a g in G. Pick upper bounds by default.
      g .= [Gi.hi for Gi in m.G[id]] # ::Vector{S}
    end
  end
  g_norm = norm(g) # ::S. ! computed with finite precision
  n=m.Model.meta.nvar # ::Int
  u = m.UList[id] #::S.
  γₙ = m.γfunc(n,u) # ::H
  ωg = H(norm(diam.(m.G[id])))/H(g_norm) * (1+γₙ)/(1-γₙ) #::H. Accounts for norm computation rounding errors, evaluated with HPFormat ≈> exact computation
  return ωg
end

@doc( @doc graderrmp!)
function graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, ::Val{REL_ERR}) where {H, S, V<:AbstractVector{S}}
  grad!(m,x,g) # ::Vector{S}
  if findfirst(x->isinf(x) || isnan(x),g) !== nothing  # one element of g overflow
    return Inf
  end
  g_norm = norm(g) #::S ! computed with finite precision in S FP format
  n=m.Model.meta.nvar # ::Int
  id = findfirst(t->t==S,m.FPList)
  u = m.UList[id] # ::H
  γₙ = m.γfunc(n,u) # ::H
  ωg = m.ωgRelErr[id] * (1+γₙ)/(1-γₙ) # ::H. Accounting for norm computation rounding errors, evaluated with HPFormat ≈> exact computation
  return ωg #::Vector{S} ::H
end

@doc( @doc graderrmp!)
function graderrmp(m::FPMPNLPModel, x::V) where {S,V<:AbstractVector{S}}
  g = similar(x)
  ωg =  graderrmp!(m, x, g)
  return g, ωg
end

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