using LinearAlgebra

import NLPModels: obj, grad, grad!, objgrad!, objgrad

export AbstractMPNLPModel,
  FPMPNLPModel,
  obj,
  grad,
  grad!,
  objerrmp,
  graderrmp,
  graderrmp!,
  objReachPrec,
  gradReachPrec!,
  AbstractMPNLPModel

const INT_ERR = 0
const REL_ERR = 1

"""
    FPMPNLPModel(Model::AbstractNLPModel{D,S},FPList::Vector{K}; kwargs...) where {D,S,K<:DataType}
    FPMPNLPModel(f,x0, FPList::Vector{DataType})

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
"""
struct FPMPNLPModel{H, B <: Tuple, D, S} <: AbstractMPNLPModel{D, S}
  Model::AbstractNLPModel{D, S}
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
  X::B
  G::B
end

function FPMPNLPModel(
  Model::AbstractNLPModel{D, S},
  FPList::Vector{K};
  HPFormat = Float64,
  γfunc = nothing,
  ωfRelErr = nothing,
  ωgRelErr = nothing,
) where {D, S, K <: DataType}
  EpsList = convert.(HPFormat, eps.(FPList))
  UList = EpsList .* HPFormat(1 / 2) # assume rounding mode is rounding to the nearest
  OFList = HPFormat.(prevfloat.(typemax.(FPList)))
  if sort(EpsList) != reverse(EpsList)
    err_msg = "FList must be ordered by increasing precision order (e.g [Float16,Float32,Float64])"
    @error err_msg
    error(err_msg)
  end
  if γfunc === nothing
    γfunc = (n::Int, u::AbstractFloat) -> n * u
  else
    γfunc_test_template(γfunc) # test provided callback function
  end
  γfunc_test_error_bound(Model.meta.nvar, EpsList[end], γfunc)

  ObjEvalMode = INT_ERR
  if ωfRelErr === nothing
    @info "Interval evaluation used by default for objective error evaluation: might significantly increase computation time"
    ObjIntervalEval_test(Model, FPList)
    ωfRelErr = Vector{HPFormat}()
  else
    @lencheck length(FPList) ωfRelErr
    ObjTypeStableTest(Model,FPList)
    ObjEvalMode = REL_ERR
  end
  GradEvalMode = INT_ERR
  if ωgRelErr === nothing
    @info "Interval evaluation used by default for gradient error evaluation: might significantly increase computation time"
    GradIntervalEval_test(Model, FPList)
    ωgRelErr = Vector{HPFormat}()
  else
    @lencheck length(FPList) ωgRelErr
    GradTypeStableTest(Model,FPList)
    GradEvalMode = REL_ERR
  end
  # instanciate interval containers X and G for point x and gradient g only if interval evaluation is used
  X = tuple()
  G = tuple()
  if ObjEvalMode == INT_ERR || GradEvalMode == INT_ERR
    X = Tuple([ElType(0) .. ElType(0) for _ = 1:(Model.meta.nvar)] for ElType in FPList)
    G = Tuple([ElType(0) .. ElType(0) for _ = 1:(Model.meta.nvar)] for ElType in FPList)
  end
  #reset counters : does not count interval evaluation of obj and grad for error test
  NLPModels.reset!(Model)
  EpsList[end] >= eps(HPFormat) || error(
    "HPFormat ($HPFormat) must be a FP format with precision equal or greater than NLPModels (max prec NLPModel: $(FPList[end]))",
  )
  EpsList[end] != eps(HPFormat) ||
    @warn "HPFormat ($HPFormat) is the same format than highest accuracy NLPModel: chances of numerical instability increased"
  FPMPNLPModel(
    Model,
    Model.meta,
    MPCounters(FPList),
    FPList,
    EpsList,
    UList,
    OFList,
    γfunc,
    ωfRelErr,
    ωgRelErr,
    ObjEvalMode,
    GradEvalMode,
    X,
    G,
  )
end

# function FPMPNLPModel(s::Symbol,
#   FPList::Vector{DataType};
#   nvar::Int=100,
#   kwargs...
# )
#   Model = eval(s)(type = Val(FPList[end]),n=nvar,gradient_backend = ADNLPModels.GenericForwardDiffADGradient)
#   FPMPNLPModel(Model,FPList;kwargs...)
# end

function FPMPNLPModel(f, x0, FPList::Vector{DataType}; nvar::Int = 100, kwargs...)
  type = eltype(x0)
  if !(type in FPList)
    error("eltype of x0 ($type) must be in FPList ($FPList)")
  end
  Model = ADNLPModel(f, x0, n = nvar, gradient_backend = ADNLPModels.GenericForwardDiffADGradient)
  FPMPNLPModel(Model, FPList; kwargs...)
end

function Base.show(io::IO, m::FPMPNLPModel)
  print(io, m.Model)
end

function NLPModels.obj(
  m::FPMPNLPModel,
  x::Union{AbstractVector{T}, AbstractVector{Interval{T}}},
) where {T <: AbstractFloat}
  increment!(m, :neval_obj, T)
  obj(m.Model, x)
end

function NLPModels.grad!(
  m::FPMPNLPModel,
  x::S,
  g::S,
) where {T <: AbstractFloat, S <: Union{AbstractVector{T}, AbstractVector{Interval{T}}}}
  increment!(m, :neval_grad, T)
  grad!(m.Model, x, g)
end

function get_id(m::FPMPNLPModel, FPFormat::DataType)
  return findfirst(t -> t == FPFormat, m.FPList)
end

"""
    objerrmp(m::FPMPNLPModel, x::AbstractVector{T})
    objerrmp(m::FPMPNLPModel, x::AbstractVector{S}, ::Val{INT_ERR})
    objerrmp(m::FPMPNLPModel, x::AbstractVector{S}, ::Val{REL_ERR})

Evaluates the objective and the evaluation error. The two functions with the extra argument ::Val{INT_ERR} and ::Val{REL_ERR} handles the interval and "classic" evaluation of the objective and the error, respectively.
Inputs: x::Vector{S}, can be either a vector of AbstractFloat or a vector of Intervals.
Outputs: fl(f(x)), ωf <: AbstractFloat, where |f(x)-fl(f(x))| ≤ ωf with fl() the floating point evaluation.
Overflow cases:
* Interval evaluation: overflow occurs if the diameter of the interval enclosing f(x) is Inf. Returns 0, Inf
* Classical evaluation:
  + If obj(x) = Inf: returns: Inf, Inf
  + If obj(x) != Inf and ωf = Inf, returns: obj(x), Inf  
"""
function objerrmp(m::FPMPNLPModel, x::AbstractVector{S}) where {S}
  get_id(m, S) !== nothing || error(
    "Floating point format of x ($S) not supported by the multiprecison model (FP formats supported: $(m.FPList))",
  )
  objerrmp(m, x, Val(m.ObjEvalMode))
end

function objerrmp(m::FPMPNLPModel, x::AbstractVector{S}, ::Val{INT_ERR}) where {S}
  id = get_id(m, S)
  for i in eachindex(x) # this is the proper way to instanciate interval vector, see issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546
    m.X[id][i] = x[i] .. x[i] # ::Vector{Interval{S}}
  end
  F = obj(m, m.X[id]) # ::Interval{S}
  if check_overflow(F) #overflow case
    return S(0.0), Inf
  else
    return mid(F), radius(F) #::S, ::S
  end
end

function objerrmp(m::FPMPNLPModel{H}, x::AbstractVector{S}, ::Val{REL_ERR}) where {H, S}
  f = obj(m, x)
  if check_overflow(f) #overflow case
    return f, Inf
  else
    id = get_id(m, S)
    ωf = H(abs(f)) * m.ωfRelErr[id] # Computed with H ≈> exact evaluation 
    return f, ωf # FP format of second returned value is H 
  end
end

"""
    graderrmp!(m::FPMPNLPModel{H}, x::V, g::V) where {H, S, V<:AbstractVector{S}}
    graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, ::Val{INT_ERR}) where {H, S, V<:AbstractVector{S}}
    graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, ::Val{REL_ERR}) where {H, S, V<:AbstractVector{S}}

Evaluates the gradient g and the relative evaluation error ωg. The two functions with the extra argument ::Val{INT_ERR} and ::Val{REL_ERR} handles the interval and "classic" evaluation of the objective and the error, respectively.
Inputs: x::Vector{S} with S in m.FPList
Outputs: g::Vector{S}, ωg <: AbstractFloat satisfying: ||∇f(x) - fl(∇f(x))||₂ ≤ ωg||g||₂ with fl() the floating point evaluation.
Note: ωg FP format may be different than S
Overflow cases:
* Interval evaluation: if at least one element of g has infinite diameter, returns [0]ⁿ, Inf
* Classical evaluation: if one element of g overflow, returns g, Inf 
"""
function graderrmp!(m::FPMPNLPModel{H}, x::V, g::V) where {H, S, V <: AbstractVector{S}}
  get_id(m, S) !== nothing || error(
    "Floating point format of x ($S) not supported by the multiprecison model (FP formats supported: $(m.FPList))",
  )
  graderrmp!(m, x, g, Val(m.GradEvalMode))
end

function graderrmp!(
  m::FPMPNLPModel{H},
  x::V,
  g::V,
  ::Val{INT_ERR},
) where {H, S, V <: AbstractVector{S}}
  id = get_id(m, S)
  for i in eachindex(x) # this is the proper way to instanciate interval vector, see issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546
    m.X[id][i] = x[i] .. x[i] # ::Vector{Interval{S}}
  end
  grad!(m, m.X[id], m.G[id]) # ::IntervalBox{S}
  if check_overflow(m.G[id])  #overflow case
    g .= zero(S)
    return Inf
  end
  g .= mid.(m.G[id]) # ::Vector{S}
  if findfirst(x -> x !== S(0), g) === nothing # g = mid(G) == 0ⁿ
    if findfirst(x -> radius(x) !== S(0), m.G[id]) === nothing # G = [0,0]ⁿ
      return 0
    else # G = [-Gᵢ,Gᵢ], pick a g in G. Pick upper bounds by default.
      g .= [Gi.hi for Gi in m.G[id]] # ::Vector{S}
    end
  end
  g_norm = norm(g) # ::S. ! computed with finite precision
  n = m.Model.meta.nvar # ::Int
  u = m.UList[id] #::S.
  γₙ = m.γfunc(n, u) # ::H
  ωg = H(norm(diam.(m.G[id]))) / H(g_norm) * (1 + γₙ) / (1 - γₙ) #::H. Accounts for norm computation rounding errors, evaluated with HPFormat ≈> exact computation
  return ωg
end

function graderrmp!(
  m::FPMPNLPModel{H},
  x::V,
  g::V,
  ::Val{REL_ERR},
) where {H, S, V <: AbstractVector{S}}
  grad!(m, x, g) # ::Vector{S}
  if check_overflow(g)
    return Inf
  end
  g_norm = norm(g) #::S ! computed with finite precision in S FP format
  n = m.meta.nvar # ::Int
  id = get_id(m, S)
  u = m.UList[id] # ::H
  γₙ = m.γfunc(n, u) # ::H
  ωg = m.ωgRelErr[id] * (1 + γₙ) / (1 - γₙ) # ::H. Accounting for norm computation rounding errors, evaluated with HPFormat ≈> exact computation
  return ωg #::Vector{S} ::H
end

@doc (@doc graderrmp!) function graderrmp(m::FPMPNLPModel, x::V) where {S, V <: AbstractVector{S}}
  g = similar(x)
  ωg = graderrmp!(m, x, g)
  return g, ωg
end

####### Objective and Grandient evaluation with prescribed error bounds ########

"""
    objReachPrec(m::FPMPNLPModel{H}, x::T, err_bound::H; π::Int = 1) where {T <: Tuple, H}

Evaluates objective and increase model precision to reach necessary error bound or to avoid overflow.
##### Inputs
* `π`: Initial ''guess'' precision level that can provide evaluation error lower than `err_bound`, use 1 by default (lowest precision)
##### Outputs
* `f`: objective value at `x`
* `ωf`: objective evaluation error
* `id`: precision level used for evaluation

There is no guarantee that `ωf ≤ err_bound`, happens if highest precision FP format is not accurate enough.
"""
function objReachPrec(m::FPMPNLPModel{H}, x::T, err_bound::H; π::Int = 1) where {T <: Tuple, H}
  id = π
  πmax = length(m.FPList)
  f, ωf = objerrmp(m, x[id])
  while ωf > err_bound && id ≤ πmax - 1
    id += 1
    f, ωf = objerrmp(m, x[id])
  end
  if id == πmax && isinf(f)
    @warn "Objective evaluation overflows with highest FP format"
  end
  if id == πmax && isinf(ωf)
    "Objective evaluation error overflows with highest FP format"
  end
  return H(f), H(ωf), id
end

"""
    gradReachPrec!(m::FPMPNLPModel{H}, x::T, g::T, err_bound::H; π::Int = 1) where {T <: Tuple, H}

Evaluates gradient and increase model precision to reach necessary error bound or to avoid overflow.
# Inputs
* `π`: Initial ''gess'' for precision level that can provide evaluation error lower than `err_bound`, uses 1 by default (lowest precision)
# Outputs
* `ωg`: objective evaluation error
* `id`: precision level used for evaluation
There is no guarantee that `ωg ≤ err_bound`. This case happens if the highest precision FP format is not accurate enough.
"""
function gradReachPrec!(
  m::FPMPNLPModel{H},
  x::T,
  g::T,
  err_bound::H;
  π::Int = 1,
) where {T <: Tuple, H}
  id = π
  πmax = length(m.FPList)
  ωg = graderrmp!(m, x[id], g[id])
  umpt!(g, g[id])
  while ωg > err_bound && id ≤ πmax - 1
    id += 1
    ωg = graderrmp!(m, x[id], g[id])
    umpt!(g, g[id])
  end
  if findfirst(x -> x == Inf, g) !== nothing
    @warn "Gradient evaluation overflows with highest FP format at x0"
  end
  if id == πmax && isinf(ωg)
    "Gradient evaluation error overflows with highest FP format at x0"
  end
  return H(ωg), id
end

@doc (@doc gradReachPrec!) function gradReachPrec(
  m::FPMPNLPModel{H},
  x::T,
  err_bound::H;
  π::Int = 1,
) where {T <: Tuple, H}
  nvar = length(x[1])
  g = Tuple(Vector{ElType}(undef, nvar) for ElType in m.FPList)
  ωg, id = gradReachPrec!(m, x, g, err_bound, π = π)
  return g, H(ωg), id
end
