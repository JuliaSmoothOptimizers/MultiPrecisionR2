using LinearAlgebra

import NLPModels: obj, grad, grad!, objgrad!, objgrad

export AbstractMPNLPModel,
  FPMPNLPModel,
  obj,
  grad,
  grad!,
  hprod!,
  hess_coord!,
  hess_structure!,
  objerrmp,
  graderrmp,
  graderrmp!,
  objReachPrec,
  gradReachPrec,
  gradReachPrec!,
  hprod_of_mp,
  hprod_of_mp!,
  hess_coord_of_mp,
  hess_coord_of_mp!,
  AbstractMPNLPModel

const INT_ERR = 0
const REL_ERR = 1

"""
    FPMPNLPModel(Model::AbstractNLPModel{D,S},FPList::Vector{K}; kwargs...) where {D,S,K<:DataType}
    FPMPNLPModel(f,x0, FPList::Vector{DataType}; kwargs...)

Floating-Point Multi-Precision Non Linear Model (`FPMPNLPModel`) structure. This structure is intended to extend `NLPModel` structure to multi-precision.

Provides errors on objective function and grandient evaluation (see `objerrmp` and `graderrmp`).

The error models are :
+ ojective: |fl(f(x)) - f(x)| ≤ ωf
+ gradient: ||fl(∇f(x)) - ∇f(x)||₂ ≤ ||fl(∇f(x))||₂ ωg.
ωf and ωg are evaluated either using:
+ interval analysis (can be very slow)
+ based on relative error assumption (see `ωfRelErr` and `ωgRelErr` field description below)   

# Fields
- `Model::AbstractNLPModel` : NLPModel
- `FPList::Vector{DataType}` : List of floating point formats
- `EpsList::Vector{H}` : List of machine epsilons of the floating point formats in `FPList`
- `UList::Vector{H}` : List of unit round-off of the floating point formats in `FPList`
- `γfunc` : callback function for dot product rounding error parameter |γ|, |fl(x.y) - x.y| ≤ |x|.|y| γ. Expected signature is `γfunc(n::Int,u::H)` and output is `H`. Default callback `γfunc(n::Int,u::H) = n*u` is implemented upon instantiation. 
- `ωfRelErr::Vector{H}` : List of relative error factor for objective function evaluation for formats in `FPList`. Error model is |f(x)-fl(f(x))| ≤ ωfRelErr * |fl(f(x))| 
- `ωgRelErr::Vector{H}` : List of relative error factor for gradient evaluation for formats in `FPList`. Error model is ||∇f(x)-fl(∇f(x))||₂ ≤ ωgRelErr * ||fl(∇f(x))||₂ 
- `ObjEvalMode::Int` : Evalutation mode for objective and error. Set automatically upon instantiation. Possible values:
  + `INT_ERR` : interval evaluation of objective (chosen as middle of the interval) and error
  + `REL_ERR` : classical evaluation and use relative error model (with `ωfRelErr` value)
- `GradEvalMode::Int` : Evalutation mode for gradient and error. Set automatically upon instantiation. Possible values:
  + `INT_ERR` : interval evaluation of gradient (chosen as middle of interval vector) and error
  + `REL_ERR` : classical evaluation and use relative error model (with `ωgRelErr` value)

# Constructors:
- `FPMPModel(Model, FPList; kwargs...)` :
  * `Model::AbstractNLPModel`: Base model
  * `FPList::Vector{DataType}`: List of FP formats that can be used for evaluations

- `FPModels(f,x0::Vector,FPList::Vector{DataType}; kwargs...)` : Instanciate a `ADNLPModel` with `f` and `x0` and call above constructor
  * `f` : objective function
  * `x0` : initial solution
  * `FPList::Vector{DataType}`: List of FP formats that can be used for evaluations

# Keyword arguments: 
  + `HPFormat=Float64` : high precision format (must be at least as accurate as FPList[end])
  + `γfunc=nothing` : use default if not provided (see Fields section above)
  + `ωfRelErr=HPFormat.(sqrt.(eps.(FPList)))`: use relative error model by default for objective evaluation
  + `ωgRelErr=HPFormat.(sqrt.(eps.(FPList)))`: use relative error model by default for gradient evaluation
  + `obj_int_eval = false` : if true, use interval arithmetic for objective value and error evaluation
  + `grad_int_eval = false` : if true, use interval arithmetic for gradient value and error evaluation
  
# Checks upon instantiation

Some checks are performed upon instantiation. These checks include:
+ Length consistency of vector fields:  FPList, EpsList
+ HPFormat is at least as accurate as the highest precision floating point format in `FPList`. Ideally HPFormat is more accurate to ensure the numerical stability of MPR2 algorithm.
+ Interval evaluations: it might happen that interval evaluation of objective function and/or gradient is type-unstable or returns an error. The constructor returns an error in this case. This type of error is most likely due to `IntervalArithmetic.jl`.
+ FPList is ordered by increasing floating point format accuracy

These checks can return `@warn` or `error`. 

# Examples 
```julia
using MultiPrecisionR2

T = [Float16, Float32]
f(x) = x[1]^2 + x[2]^2
x = zeros(2)
mpnlp = FPMPNLPModel(f,x0,T)
```
-----------
```julia
using MultiPrecisionR2
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using ADNLPModels
using BFloat16s

T = [BFloat16, Float16, Float32]
nlp = woods()
mpnlp = (nlp,T)
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
  ωfRelErr = HPFormat.(sqrt.(eps.(FPList))),
  ωgRelErr = HPFormat.(sqrt.(eps.(FPList))),
  obj_int_eval = false,
  grad_int_eval = false
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

  if obj_int_eval
    ObjEvalMode = INT_ERR
    @info "Interval evaluation used for objective error evaluation: might significantly increase computation time"
    ObjIntervalEval_test(Model, FPList)
  else
    @lencheck length(FPList) ωfRelErr
    ObjTypeStableTest(Model, FPList)
    ObjEvalMode = REL_ERR
    @info "Using relative error model for objective evaluation."
  end
  
  if grad_int_eval
    GradEvalMode = INT_ERR
    @info "Interval evaluation used for gradient error evaluation: might significantly increase computation time"
    GradIntervalEval_test(Model, FPList)
  else
    @lencheck length(FPList) ωgRelErr
    GradTypeStableTest(Model, FPList)
    GradEvalMode = REL_ERR
    @info "Using relative error model for gradient evaluation."
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

function FPMPNLPModel(f, x0, FPList::Vector{DataType}; kwargs...)
  type = eltype(x0)
  if !(type in FPList)
    error("eltype of x0 ($type) must be in FPList ($FPList)")
  end
  Model = ADNLPModel(f, x0, backend = :generic)
  FPMPNLPModel(Model, FPList; kwargs...)
end

function Base.show(io::IO, m::FPMPNLPModel)
  print(io, m.Model)
end

function NLPModels.obj(
  m::FPMPNLPModel,
  x::Union{AbstractVector{T}, AbstractVector{Interval{T}}},
) where {T <: AbstractFloat}
  MultiPrecisionR2.increment!(m, :neval_obj, T)
  obj(m.Model, x)
end

function NLPModels.grad!(
  m::FPMPNLPModel,
  x::S,
  g::S,
) where {T <: AbstractFloat, S <: Union{AbstractVector{T}, AbstractVector{Interval{T}}}}
  MultiPrecisionR2.increment!(m, :neval_grad, T)
  grad!(m.Model, x, g)
end

function NLPModels.hprod!(
  m::FPMPNLPModel,
  x::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T}
  increment!(m, :neval_hprod,T)
  hprod!(m.Model,x,v,Hv,obj_weight = obj_weight)
end

function NLPModels.hprod!(
  m::FPMPNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  v::AbstractVector{T},
  Hv::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T}
  increment!(m, :neval_hprod,T)
  hprod!(m.Model,x,y,v,Hv,obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  m::FPMPNLPModel,
  x::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T}
  @lencheck m.meta.nvar x
  @lencheck m.meta.nnzh vals
  hess_coord!(m,x,zeros(T, m.meta.ncon),vals,obj_weight = obj_weight)
end

function NLPModels.hess_coord!(
  m::FPMPNLPModel,
  x::AbstractVector{T},
  y::AbstractVector{T},
  vals::AbstractVector{T};
  obj_weight::Real = one(T),
) where {T}
  increment!(m, :neval_hess, T)
  hess_coord!(m.Model,x,y,vals,obj_weight = obj_weight)
end

function NLPModels.hess_structure!(
  m::FPMPNLPModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  hess_structure!(m.Model,rows,cols)
end

"""
    get_id(m::FPMPNLPModel, FPFormat::DataType)

Returns the index of `FPFormat` `in m.FPList`. 
"""
function get_id(m::FPMPNLPModel, FPFormat::DataType)
  return findfirst(t -> t == FPFormat, m.FPList)
end

"""
    objerrmp(m::FPMPNLPModel, x::AbstractVector{T})
    objerrmp(m::FPMPNLPModel, x::AbstractVector{S}, ::Val{INT_ERR})
    objerrmp(m::FPMPNLPModel, x::AbstractVector{S}, ::Val{REL_ERR})

Evaluates the objective and the evaluation error. The two functions with the extra argument `::Val{INT_ERR}` and `::Val{REL_ERR}` handles the interval and "classic" evaluation of the objective and the error, respectively.

# Arguments
* `x::Vector{S}`: where to evaluate the objective, can be either a vector of `AbstractFloat` or a vector of `Intervals`.

# Outputs
1. fl(f(x)): finite-precision evaluation of the objective at `x`
2. `ωf <: AbstractFloat`: evaluation error, |f(x)-fl(f(x))| ≤ ωf with fl() the floating point evaluation.

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
    graderrmp(m::FPMPNLPModel, x::V)
    graderrmp!(m::FPMPNLPModel{H}, x::V, g::V) where {H, S, V<:AbstractVector{S}}
    graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, ::Val{INT_ERR}) where {H, S, V<:AbstractVector{S}}
    graderrmp!(m::FPMPNLPModel{H}, x::V, g::V, ::Val{REL_ERR}) where {H, S, V<:AbstractVector{S}}

Evaluates the gradient g and the relative evaluation error ωg. The two functions with the extra argument `::Val{INT_ERR}` and `::Val{REL_ERR}` handles the interval and "classic" evaluation of the objective and the error, respectively.
# Arguments
- `m::FPMPNLPModel` : multi-precision model
- `x::V`: where the gradient is evaluated
- `g::V`: container for gradient

# Outputs
1. `g::Vector{S}`: gradient value, only returned with `graderrmp`.
2. `ωg <: AbstractFloat`: evaluation error satisfying: ||∇f(x) - fl(∇f(x))||₂ ≤ ωg||g||₂ with fl() the floating point evaluation.
Note: ωg FP format may be different than `S`

# Modified
* `g::V`: updated with gradient value

Overflow cases:
* Interval evaluation: if at least one element of `g` has infinite diameter, returns [0]ⁿ, Inf
* Classical evaluation: if at least one element of `g` overflows, returns `g, Inf` 
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
  n = m.meta.nvar # ::Int
  id = get_id(m, S)
  u = m.UList[id] # ::H
  γₙ = m.γfunc(n, u) # ::H
  ωg = m.ωgRelErr[id]
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
# Arguments
  * `m::FPMPNLPModel{H}`: multi precision model
  * `x::T`: tuple containing value of x in the FP formats of `FPList`
  * `err_bound::H` : evaluation error tolerance
  * `π::Int`: Initial ''guess'' for FP format that can provide evaluation error lower than `err_bound`, use 1 by default (lowest precision FP format)

# Outputs
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
    gradReachPrec(m::FPMPNLPModel{H}, x::T, err_bound::H; π::Int = 1) where {T <: Tuple, H}

Evaluates gradient and increase model precision to reach necessary error bound or to avoid overflow.
# Arguments
* `m::FPMPNLPModel{H}`: multi precision model
* `x::T`: tuple containing value of x in the FP formats of `FPList`
* `g::T`: gradient container
* `π`: Initial ''guess'' precision level that can provide evaluation error lower than `err_bound`, use 1 by default (lowest precision)
# Outputs
1. (`g`): gradient, returned only with `gradReachPrec` call  
2. `ωg`: objective evaluation error
3. `id`: precision level used for evaluation

# Modified 
* (`g`): updated with the gradient value, only with `gradReachPrec!` call

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
  if check_overflow(g[end])
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

"""
    hprod_of_mp(m::FPMPNLPModel, x::T, v::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}
    hprod_of_mp(m::FPMPNLPModel, x::T, y::T, v::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}
    hprod_of_mp!(m::FPMPNLPModel, x::T, v::T, Hv::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}
    hprod_of_mp!(m::FPMPNLPModel, x::T, y::T, v::T, Hv::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}

Call `hprod!` recursively from the π-th element of the tuple arguments until `Hv` does not overflow.

# Modified argument
* `Hv::T`

# Outputs
1. `Hv::T`: only returned with `hprod_of_mp`
2. `id::Int` : index of updated `Hv` element

"""
function hprod_of_mp!(
  m::FPMPNLPModel,
  x::T,
  y::T,
  v::T,
  Hv::T;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  id = π
  πmax = length(m.FPList)
  hprod!(m,x[id],y[id],v[id],Hv[id],obj_weight = m.FPList[id](obj_weight))
  while check_overflow(Hv[id]) && id <= πmax -1
    id += 1
    hprod!(m,x[id],y[id],v[id],Hv[id],obj_weight = m.FPList[id](obj_weight))
  end
  umpt!(Hv, Hv[id])
  return id
end

function hprod_of_mp!(
  m::FPMPNLPModel,
  x::T,
  v::T,
  Hv::T;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  hprod_of_mp!(m,x,Tuple(zeros(t, m.meta.ncon) for t in m.FPList),v,Hv,obj_weight = obj_weight,π=π)
end

function hprod_of_mp(
  m::FPMPNLPModel,
  x::T,
  y::T,
  v::T;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  Hv = Tuple(similar(x[i]) for i in eachindex(x))
  id = hprod_of_mp!(m,x,y,v,Hv,obj_weight = obj_weight,π = π)
  return Hv, id
end

function hprod_of_mp(
  m::FPMPNLPModel,
  x::T,
  v::T;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  Hv = Tuple(similar(x[i]) for i in eachindex(x))
  id = hprod_of_mp!(m,x,v,Hv,obj_weight = obj_weight,π = π)
  return Hv, id
end

"""
    hess_coord_of_mp(m::FPMPNLPModel, x::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}
    hess_coord_of_mp(m::FPMPNLPModel, x::T, y::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}
    hess_coord_of_mp!(m::FPMPNLPModel, x::T, vals::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}
    hess_coord_of_mp!(m::FPMPNLPModel, x::T, y::T, vals::T; obj_weight::Real = 1.0, π::Int = 1) where {T <: Tuple}

Call `hess_coord!` recursively from the π-th element of the tuple arguments until `vals` does not overflow.

# Modified argument
* `vals::T`

# Outputs
1. `vals::T`: only returned with `hess_coord_of_mp`
2. `id::Int` : index of updated `vals` element

"""
function hess_coord_of_mp!(
  m::FPMPNLPModel,
  x::T,
  y::T,
  vals::T,;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  id = π
  πmax = length(m.FPList)
  hprod!(m,x[id],y[id],vals[id],obj_weight = m.FPList[id](obj_weight))
  while check_overflow(Hv[id]) && id <= πmax -1
    id += 1
    hprod!(m,x[id],y[id],vals[id],obj_weight = m.FPList[id](obj_weight))
  end
  umpt!(Hv, Hv[id])
  return id
end

function hess_coord_of_mp!(
  m::FPMPNLPModel,
  x::T,
  vals::T;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  hess_coord_of_mp!(m,x,Tuple(zeros(t, m.meta.ncon) for t in m.FPList),vals,obj_weight = obj_weight,π=π)
end

function hess_coord_of_mp(
  m::FPMPNLPModel,
  x::T,
  y::T;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  vals = Tuple(Vector{t}(undef,m.meta.nnzh) for t in m.FPList)
  id = hess_coord_of_mp!(m,x,y,vals,obj_weight = obj_weight,π = π)
  return vals, id
end

function hess_coord_of_mp(
  m::FPMPNLPModel,
  x::T;
  obj_weight::Real = 1.0,
  π::Int = 1
) where {T <: Tuple}
  vals = Tuple(Vector{t}(undef,m.meta.nnzh) for t in m.FPList)
  id = hess_coord_of_mp!(m,x,vals,obj_weight = obj_weight,π = π)
  return vals, id
end