using ForwardDiff
using LinearAlgebra

"""
Extends ADNLPModels.grad to IntervalBox argument.
"""
function NLPModels.grad(nlp::ADNLPModel,X::Vector{Interval{S}}) where S
  G = similar(X)
  grad!(nlp,X,G)
  return G
end

function NLPModels.grad!(nlp::ADNLPModel,X::Vector{Interval{S}},G::Vector{Interval{S}}) where S
  nlp.counters.neval_grad+=1
  ForwardDiff.gradient!(G,nlp.f,X)
end

NLPModels.grad!
const INT_ERR = 0
const REL_ERR = 1

abstract type AbstractMPNLPModel end

"""
Floating-Point Multi-Precision Non Linear Model structure. This structure is intended to be used as MPR2Solver input.

Primairly stores NLPmodels instanciated with different Floating Point formats and provide errors on objective function and grandient evaluation (see `objerrmp` and `graderrmp`).
The error models are :
+ ojective: |fl(f(x)) - f(x)| ≤ ωf
+ gradient: |fl(∇f(x)) - ∇f(x)| ≤ ||fl(∇f(x))||₂ ωg.
ωf and ωg are needed for MPR2Solver. They are evaluated either using:
+ interval analysis (can be very slow)
+ based on relative error assumption (see `ωfRelErr` and `ωgRelErr` field description below)   


# Fields
- `MList::AbstractVector` : List of NLPModels, ordered by increasing precision floating point formats
- `FPList::Vector{DataType}` : List of floating point formats of the NLPModels in MList
- `EpsList::Vector{H}` : List of machine epsilons of the floating point formats
- `HPFormat::DataType` : High precision floating point format, used for intermediate value computation in MPR2Solver. Is H parameter.
- `γfunc` : callback function for dot product rounding error parameter |γ|, |fl(x.y) - x.y| ≤ |x|.|y| γ. Expected signature is `γfunc(n::Int,u::H)` and output is `H`. Default callback `γfunc(n::Int,u::H) = n*u` is implemented upon instanciation. 
- `ωfRelErr::Vector{H}` : List of relative error factor for objective function evaluation of the MList NLPModels. Error model is |obj.(MList[i],x)-fl(f(x))| ≤ ωfRelErr[i] * |obj.(MList[i],x)| 
- `ωgRelErr::Vector{H}` : List of relative error factor for gradient of the MList NLPModels. Error model is |grad.(MList[i],x)-fl(∇f(x))| ≤ ωgRelErr[i] * ||grad.(MList[i],x)||₂ 
- `ObjEvalMode::Int` : Evalutation mode for objective and error. Set automatically upon instanciation. Possible values:
  + `INT_ERR` : interval evaluation of objective (chosen as middle of the interval) and error
  + `REL_ERR` : classical evaluation and use relative error model (with ωfRelErr value)
- `GradEvalMode::Int` : Evalutation mode for gradient and error. Set automatically upon instanciation. Possible values:
  + `INT_ERR` : interval evaluation of gradient (chosen as middle of interval vector) and error
  + `REL_ERR` : classical evaluation and use relative error model (with ωgRelErr value)

# Constructors:
- `FPMPModel(MList::AbstractVector{M}; kwargs...)` where M <:NLPModel : create a FPMPModel with MList a list of NLPModels
- `FPMPModel(FPList::Vector{DataType}; nvar::Int=100, kwargs...)` : create a FPMPModel from a symbol linked to a NLPModel.

Keyword arguments: 
- nvar: dimension of the problem (if scalable)
- kwargs: 
  + `HPFormat=Float64` : high precision format (must be at least as accurate as FPList[end])
  + `γfunc=nothing` : use default if not provided (see Fields section above)
  + `ωfRelErr=nothing` : use interval evaluation if not provided
  + `ωgRelErr=nothing` : use interval evaluation if not provided

# Checks upon instanciation
Some checks are performed upon instanciation. These checks include:
+ Length consistency of vector fields: MList, FPList, EpsList
+ HPFormat is at least as accurate as the highest precision floating point format among the ones of the NLPModels in MList. Ideally HPFormat is more accurate to ensure the numerical stability of MPR2 algorithm.
+ Interval evaluations: it might happen that interval evaluation of objective function and/or gradient is type-unstable or returns an error. The constructor returns an error in this case. This type of error is most likely due to `IntervalArithmetic.jl`.
+ NLPModels of MList are order by increasing floating point format accuracy

This checks can return `@warn` or `error`. 

# Examples 
```julia
T = [Float16, Float32]
f(x) = x[1]^2 + x[2]^2
x = zeros(2)
MList = [ADNLPModel(f,t.(x)) for t in T];
MPmodel = FPMPNLPModel(MList)
```

```julia
T = [Float16, Float32]
problems = setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
s = problem[end]
MPmodel = FPMPNLPModel(s,T)
```
"""
struct FPMPNLPModel{H,F,T<:Tuple} <: AbstractMPNLPModel
  MList::AbstractVector
  FPList::Vector{DataType}
  EpsList::Vector{H}
  UList::Vector{H}
  OFList::Vector{H}
  γfunc::F
  ωfRelErr::Vector{H}
  ωgRelErr::Vector{H}
  ObjEvalMode::Int
  GradEvalMode::Int
  X::T
  G::T
end

function FPMPNLPModel(MList::AbstractVector{M};
  HPFormat=Float64,
  γfunc=nothing,
  ωfRelErr=nothing,
  ωgRelErr=nothing
) where {M<:AbstractNLPModel}
  nlpdim_test(MList)
  FPList = [typeof(nlp.meta.x0[1]) for nlp in MList]
  EpsList = convert.(HPFormat,eps.(FPList))
  UList = EpsList.*HPFormat(1/2) # assume rounding mode is rounding to the nearest
  OFList = HPFormat.(prevfloat.(typemax.(FPList)))
  if sort(EpsList) != reverse(EpsList)
    err_msg = "MList: wrong models FP formats precision order"
    @error err_msg
    error(err_msg)
  end
  if γfunc === nothing 
    γfunc = (n::Int,u::AbstractFloat) -> n*u
  else
    γfunc_test_template(γfunc) # test provided callback function
  end
  γfunc_test_error_bound(MList[1].meta.nvar,EpsList[end],γfunc)

  ObjEvalMode = INT_ERR
  if ωfRelErr === nothing
    @info "Interval evaluation used by default for objective error evaluation: might significantly increase computation time"
    ObjIntervalEval_test(MList)
    ωfRelErr=Vector{HPFormat}()
  else
    fRelList_test(MList,ωfRelErr)
    ObjEvalMode = REL_ERR
  end
  GradEvalMode = INT_ERR
  if ωgRelErr === nothing
    @info "Interval evaluation used by default for gradient error evaluation: might significantly increase computation time"
    GradIntervalEval_test(MList)
    ωgRelErr=Vector{HPFormat}()
  else
    gRelList_test(MList,ωgRelErr)
    GradEvalMode = REL_ERR
  end
  # instanciate interval containers X and G for point x and gradient g only if interval evaluation is used
  X = tuple()
  G = tuple()
  if ObjEvalMode == INT_ERR || GradEvalMode == INT_ERR
    setrounding(Interval,:accurate)
    X = Tuple([ElType(0) .. ElType(0) for _ in 1:MList[1].meta.nvar] for ElType in FPList)
    G = Tuple([ElType(0) .. ElType(0) for _ in 1:MList[1].meta.nvar] for ElType in FPList)
  end
  #reset counters : does not count interval evaluation of obj and grad for error test
  for nlp in MList
    reset!(nlp)
  end
  EpsList[end] >= eps(HPFormat) || error("HPFormat ($HPFormat) must be a FP format with precision equal or greater than NLPModels (max prec NLPModel: $(FPList[end]))")
  EpsList[end] != eps(HPFormat) || @warn "HPFormat ($HPFormat) is the same format than highest accuracy NLPModel: chances of numerical instability increased"
  FPMPNLPModel(MList,FPList,EpsList,UList,OFList,γfunc,ωfRelErr,ωgRelErr,ObjEvalMode,GradEvalMode,X,G)
end

function FPMPNLPModel(s::Symbol,
  FPList::Vector{DataType};
  nvar::Int=100,
  kwargs...
)
  MList=Vector{AbstractNLPModel}()
  for fp in FPList
    nlp = eval(s)(type = Val(fp),n=nvar)
    push!(MList,nlp)
  end
  FPMPNLPModel(MList,kwargs...)
end

"""
Evaluate the objective function of the id-th model of m.
Format of the returned value is m.FPList[id].
""" 
function objmp(m::FPMPNLPModel,x::V,id::Int) where {S,V<:AbstractVector{S}}
  S == m.FPList[id] || error("Expected input format $(m.FPList[id]) for x but got $S")
  obj(m.MList[id],x)
end

"""
Evaluate the objective and the evaluation error of the id-th model of m.
Inputs: x::Vector{S} with S == m.FPFormat[id]
Outputs: ̂f::S, ωf <: AbstractFloat, with |f(x)-̂f| ≤ ωf
Overflow cases:
* Interval evaluation: overflow occurs if the diameter of the interval enclosing f(x) is Inf. Returns 0, Inf
* Classical evaluation:
  + If obj(x) = Inf: returns: Inf, Inf
  + If obj(x) != Inf and ωf = Inf, returns: obj(x), Inf  
"""
function objerrmp(m::FPMPNLPModel, x::V, id::Int) where {S,V<:AbstractVector{S}}
  S == m.FPList[id] || error("Expected input format $(m.FPList[id]) for x but got $S")
  objerrmp(m, x, id, Val(m.ObjEvalMode))
end

@doc (@doc objerrmp)
function objerrmp(m::FPMPNLPModel, x::V, id::Int, ::Val{INT_ERR}) where {S,V<:AbstractVector{S}}
  for i in eachindex(x) # this is the proper way to instanciate interval vector, see issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546
    m.X[id][i] = x[i] .. x[i] # ::Vector{Interval{S}}
  end
  F = obj(m.MList[id],m.X[id]) # ::Interval{S}
  if diam(F) == Inf #overflow case
    return S(0.0), Inf
  else
    return mid(F), radius(F) #::S, ::S
  end
end

@doc (@doc objerrmp)
function objerrmp(m::FPMPNLPModel{H}, x::V, id::Int, ::Val{REL_ERR}) where {H, S, V<:AbstractVector{S}}
  f = obj(m.MList[id],x) #:: S
  if f === S(Inf) #overflow case
    return f, Inf
  else
    ωf = H(abs(f))*m.ωfRelErr[id] # Computed with H ≈> exact evaluation 
    return f, ωf # FP format of second returned value is H 
  end
end
"""
Evaluate the gradient of the id-th model of m.
""" 
function gradmp(m::FPMPNLPModel, x::V, id::Int) where {S, V<:AbstractVector{S}}
  S == m.FPList[id] || error("Expected input format $(m.FPList[id]) for x but got $S")
  grad(m.MList[id],x)
end
 
@doc (@doc gradmp) function gradmp!(m::FPMPNLPModel, x::V, id::Int, g::V) where {S, V<:AbstractVector{S}}
  S == m.FPList[id] || error("Expected input format $(m.FPList[id]) for x but got $S")
  grad!(m.MList[id],x,g)
end

"""
Evaluate the gradient g and the relative evaluation error ωg of the id-th model of m.
Inputs: x::Vector{S} with S == m.FPFormat[id]
Outputs: g::Vector{S}, ωg <: AbstractFloat satisfying: ||∇f(x) - g||₂ ≤ ωg||g||₂
Note: ωg FP format may be different than S
Overflow cases:
* Interval evaluation: if at least one element of g has infinite diameter, returns [0]ⁿ, Inf
* Classical evaluation: if one element of g overflow, returns g, Inf 
"""
function graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, id::Int) where {H, S, V<:AbstractVector{S}}
  S == m.FPList[id] || error("Expected input format $(m.FPList[id]) for x but got $S")
  graderrmp!(m, x, g, id, Val(m.GradEvalMode))
end

@doc( @doc graderrmp!)
function graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, id::Int, ::Val{INT_ERR}) where {H, S, V<:AbstractVector{S}}
  for i in eachindex(x) # this is the proper way to instanciate interval vector, see issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546
    m.X[id][i] = x[i] .. x[i] # ::Vector{Interval{S}}
  end
  grad!(m.MList[id],m.X[id],m.G[id]) # ::IntervalBox{S}
  if findfirst(x->diam(x) === S(Inf),m.G[id]) !== nothing  #overflow case
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
  n=m.MList[1].meta.nvar # ::Int
  u = m.UList[id] #::S.
  γₙ = m.γfunc(n,u) # ::H
  ωg = H(norm(diam.(m.G[id])))/H(g_norm) * (1+γₙ)/(1-γₙ) #::H. Accounts for norm computation rounding errors, evaluated with HPFormat ≈> exact computation
  return ωg
end

@doc( @doc graderrmp!)
function graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, id::Int, ::Val{REL_ERR}) where {H, S, V<:AbstractVector{S}}
  grad!(m.MList[id],x,g) # ::Vector{S}
  if findfirst(x->x===S(Inf),g) !== nothing # one element of g overflow
    return Inf
  end
  g_norm = norm(g) #::S ! computed with finite precision in S FP format
  n=m.MList[1].meta.nvar # ::Int
  u = m.UList[id] # ::H
  γₙ = m.γfunc(n,u) # ::H
  ωg = m.ωgRelErr[id] * (1+γₙ)/(1-γₙ) # ::H. Accounting for norm computation rounding errors, evaluated with HPFormat ≈> exact computation
  return ωg #::Vector{S} ::H
end

@doc( @doc graderrmp!)
function graderrmp(m::FPMPNLPModel, x::V, id::Int) where {S,V<:AbstractVector{S}}
  g = similar(x)
  ωg =  graderrmp!(m, x, g, id)
  return g, ωg
end

"""
Returns the  list of counters of the NLPModels in MList.
See [`FPMPNLPModel`](@ref)
"""
function get_counters(MPnlp::FPMPNLPModel)
  return [m.counters for m in MPnlp.MList]
end

"""
Test interval evaluation of objective of all the AbstractNLPModels in MList.
Test fails and return an error if:
  * Interval evaluation returns an error
  * Interval evaluation is not type stable
See [`FPMPNLPModel`](@ref), [`AbstractNLPModel`](@ref)
"""
function ObjIntervalEval_test(MList::AbstractVector{M}) where M<:AbstractNLPModel
  i=1
  for nlp in MList
    @debug "Testing objective evaluation of MList[$i]"
    try
      X0 = [xi..xi for xi ∈ nlp.meta.x0]
      intype = typeof(nlp.meta.x0[1])
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
  i+=1
end

"""
Test interval evaluation of gradient of all the AbstractNLPModels in MList.
Test fails and return an error if:
  * Interval evaluation returns an error
  * Interval evaluation is not type stable
See [`FPMPNLPModel`](@ref), [`AbstractNLPModel`](@ref)
"""
function GradIntervalEval_test(MList::AbstractVector{M}) where M<:AbstractNLPModel
  i=1
  for nlp in MList
    @debug "Testing grad evaluation of MList[$i]"
    try 
      # X0 = IntervalBox(nlp.meta.x0)
      X0 = [xi..xi for xi ∈ nlp.meta.x0]
      intype = typeof(nlp.meta.x0[1])
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
    i+=1
  end
end

"""
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
Tests dimensions match of NLPModels in FPMPModel.MList.
"""
function nlpdim_test(MList::Vector{M}) where M<:AbstractNLPModel
  l = length(MList)
  if l>1
    err_msg = "FPMPModel.MList NLPModels dimensions mismatch"
    nvar = MList[1].meta.nvar
    sum([MList[i].meta.nvar != nvar for i=2:length(MList)]) == 0 || error(err_msg)
  end
end

"""
Tests dimensions match of MList and ωfRelErr
"""
function fRelList_test(MList::Vector{M}, ωfRelErr::Vector{H}) where {H,M<:AbstractNLPModel}
  err_msg = "MList and ωfRelErr dimension mismatch"
  length(MList)==length(ωfRelErr) || error(err_msg)
  end

"""
Tests dimensions match of MList and ωgRelErr
"""
function gRelList_test(MList::Vector{M}, ωgRelErr::Vector{H}) where {H,M<:AbstractNLPModel}
  err_msg = "MList and ωgRelErr dimension mismatch"
  length(MList)==length(ωgRelErr) || error(err_msg)
end