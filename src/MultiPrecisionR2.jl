# Author: D. Monnet

# Multi precision regularization algorithm.

# This work has been supported by the NSERC Alliance grant 544900- 19 in collaboration with Huawei-Canada. 

module MultiPrecisionR2

using ADNLPModels, IntervalArithmetic, NLPModels, Printf, LinearAlgebra, SolverCore

export MPR2, MPR2Solver, MPR2Params, MPR2State, MPR2Precisions, solve!, umpt!, update_struct!

abstract type AbstractMPNLPModel{T,S} <: AbstractNLPModel{T,S} end

include("MPCounters.jl")
include("MPNLPModels.jl")
include("utils.jl")

"""
    MPR2(MPnlp; kwargs...)
An implementation of the quadratic regularization algorithm with dynamic selection of floating point format for objective and gradient evaluation, robust against finite precision rounding errors.

# Arguments
- `MPnlp::FPMPNLPModel{H}` : Multi precision model, see `FPMPNLPModel`
Keyword agruments:
- `x₀::V = MPnlp.Model.meta.x0` : initial guess 
- `par::MPR2Params = MPR2Params(MPnlp.FPList[1],H)` : MPR2 parameters, see `MPR2Params` for details
- `atol::H = H(sqrt(eps(T)))` : absolute tolerance on first order criterion 
- `rtol::H = H(sqrt(eps(T)))` : relative tolerance on first order criterion
- `max_eval::Int = -1`: maximum number of evaluation of the objective function.
- `max_iter::Int = 1000` : maximum number of iteration allowed
- `σmin::T = sqrt(T(MPnlp.EpsList[end]))` : minimal value for regularization parameter. Value must be representable in any of the floating point formats of MPnlp. 
- `verbose::Int=0` : display iteration information if > 0
- `e::E` : user defined structure, used as argument for `compute_f_at_x!`, `compute_f_at_c!` `compute_g!` and `recompute_g!` callback functions.
- `compute_f_at_x!` : callback function to select precision and compute objective value and error bound at the current point. Allows to reevaluate the objective at x if more precision is needed.
- `compute_f_at_c!` : callback function to select precision and compute objective value and error bound at candidate.
- `compute_g!` : callback function to select precision and compute gradient value and error bound. Called at the end of main loop.
- `recompute_g!` : callback function to select precision and recompute gradient value if more precision is needed. Called after step, candidate and model decrease computation in main loop.
- `selectPic!` : callback function to select FP format of `c` at the next iteration
# Outputs
Returns a `GenericExecutionStats`, see `SolverCore.jl`
`GenericExecutionStats.status` is set to `:exception` 
# Example
```julia
T = [Float16, Float32]
f(x) = x[1]^2+x[2]^2
x = ones(2)
nlp_list = [ADNLPModel(f,t.(x)) for t in T ]
MPnlp = FPMPNLPModel(nlp_list)
mpr2s(MPnlp) 
```
"""
mutable struct MPR2Solver{T <: Tuple} <: AbstractOptimizationSolver
  x::T
  g::T
  s::T
  c::T
end

function MPR2Solver(MPnlp::M) where { M <: FPMPNLPModel }
  nvar = MPnlp.meta.nvar
  x = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  s = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  c = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  g = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  return MPR2Solver(x, s, c, g)
end

@doc (@doc MPR2Solver) function MPR2(
  MPnlp::FPMPNLPModel;
  x₀::V = MPnlp.meta.x0,
  kwargs...
) where V
  solver = MPR2Solver(MPnlp)
  return SolverCore.solve!(solver, MPnlp;x₀ = x₀, kwargs...)
end


"""
    MPR2Params(LPFormat::DataType, HPFormat::DataType)
MPR2 parameters.

# Fields
- `η₀::H` : controls objective function error tolerance, convergence condition is ωf ≤ η₀ ΔT (see `FPMPNLPModel` for details on ωf)
- `η₁::H` : step successful if ρ ≥ η₁ (update incumbent)
- `η₂::H` : step very successful if ρ ≥ η₂ (decrease σ ⟹ increase step length)
- `κₘ::H` : tolerance on gradient evaluation error, μ ≤ κₘ (see `computeMu`) 
- `γ₁::L` : σk+1 = σk * γ₁ if ρ ≥ η₂
- `γ₂::L` : σk+1 = σk * γ₂ if ρ < η₁

# Parameters
- `H` must correspond to `MPnlp.HPFormat` with `MPnlp` given as input of `MPR2`
- `L` must correspond to `MPnlp.FPList[1]`, i.e the lowest precision floating point format used by `MPnlp` given as input of `MPR2`

# Conditions 
Parameters must statisfy the following conditions:
* 0 ≤ η₀ ≤ 1/2*η₁
* 0 ≤ η₁ ≤ η₂ < 1
* η₀+κₘ/2 ≤0.5*(1-η₂)
* η₂<1 
* 0<γ₁<1<γ₂

Instiates default values:
- `η₀::H = 0.01`
- `η₁::H = 0.02`
- `η₂::H = 0.95`
- `κₘ::H = 0.02` 
- `γ₁::L = 2^(-2)`
- `γ₂::L = 2`
"""
mutable struct MPR2Params{H, L}
  η₀::H
  η₁::H
  η₂::H
  κₘ::H
  γ₁::L
  γ₂::L
end

function MPR2Params(LPFormat::DataType, HPFormat::DataType)
  η₀ = HPFormat(0.01)
  η₁ = HPFormat(0.02)
  η₂ = HPFormat(0.95)
  κₘ = HPFormat(0.02)
  γ₁ = LPFormat(2^(-2))
  γ₂ = LPFormat(2)
  return MPR2Params(η₀, η₁, η₂, κₘ, γ₁, γ₂)
end

"""
    CheckMPR2ParamConditions(p::MPR2Params{H})

Check if the MPR2 parameters conditions are satified.
See [`MPR2Params`](@ref) for parameter conditions.
"""
function CheckMPR2ParamConditions(p::MPR2Params{H}) where{H}
  0 ≤ p.η₀ ≤ 1/2*p.η₁       || error("Expected 0 ≤ η₀ ≤ 1/2*η₁")
  p.η₁ ≤ p.η₂ < 1           || error("Expected η₁ ≤ η₂ < 1")
  p.η₀+p.κₘ/2 ≤0.5*(1-p.η₂) || error("Expected η₀+κₘ/2 ≤0.5*(1-η₂)")
  p.η₂<1                    || error("Expected η₂<1")
  0<p.γ₁<1<p.γ₂             || error("Expected 0<γ₁<1<γ₂")
end
 
"""
    function MPR2Precisions(π::Int)

Precision  of variables and precision evaluation of obj, grad, model reduction and norms.
Precisions are represented by integers, and correspond to FP format of corresponding index in `FPMPNLPModel.FPList`. i.e., precision `i` correpsonds to FP format `FPMPNLPModel.FPList[i]`
See `FPMPNLPModel`.
"""
mutable struct MPR2Precisions
  πx::Int
  πnx::Int
  πs::Int
  πns::Int
  πc::Int
  πf::Int
  πf⁺::Int
  πg::Int
  πΔ::Int
end

@doc (@doc MPR2Precisions) function MPR2Precisions(π::Int)
  return MPR2Precisions([π for _ in eachindex(fieldnames(MPR2Precisions))]...)
end

Base.copy(π::MPR2Precisions) = MPR2Precisions(π.πx, π.πnx, π.πs, π.πns, π.πc, π.πf, π.πf⁺, π.πg, π.πΔ)

"""
    update_struct!(str,other_str)

Update the fields of `str` with the fields of `other_str`.
"""
function update_struct!(str::MPR2Precisions,other_str::MPR2Precisions)
  for f in fieldnames(MPR2Precisions)
    setfield!(str,f,getfield(other_str,f))
  end
end

"""
    MPR2State(HPFormat::DataType)

Intermediate variables used by MPR2 solver. This structure stores the "state" of the algorithm, can be used in callback functions to select evaluation precision.
## Fields
* x_norm: norm of current point x (estimated with FP computation with πx precision)
* s_norm: norm of step s (estimated with FP computation with πs precision)
* g_norm: norm of gradient g (estimated with FP computation with πg precision)
* ΔT: model decrease (estimated with FP computation with πΔ precision)
* ρ: success indicator (ρ = (f(x) - f(c))/ΔT)
* ϕ: upper bound on ||x||/||s||, takes norm computation error into account
* ϕhat: computed value of ||x||/||s||
* μ: indicator including gradient error ωg and finite-precision errors due to step, candidate and model decrease computation
* f: objective function value at xk, f(xk)
* f⁺: objective function value at ck, f(ck)
* ωf: objective function evaluation error at xk, |f(xk) - f| ≤ ωf(xk)
* ωf⁺: objective function evaluation error at ck, |f(xk) - f⁺| ≤ ωf(ck)
* ωg: gradient error norm bound at xk, ||g-∇f(xk)|| ≤ ωg ||g||
* ωfBound: upper bound on ωf and ωf⁺ required to ensure convergence
* σ: regularization parameter
* iter: iteration number
* status: algorithm exit status (see `GenericExecutionStats`)

See also `MPR2Precisions`, `GenericExecutionStats`.
"""
mutable struct MPR2State{H}
  x_norm::H
  s_norm::H
  g_norm::H
  ΔT::H
  ρ::H
  ϕ::H
  ϕhat::H
  μ::H
  f::H
  f⁺::H
  ωf::H
  ωf⁺::H
  ωg::H
  ωfBound::H
  σ::H
  iter::Int
  status
end

function MPR2State(HPFormat::DataType)
  return MPR2State([HPFormat(0) for _ in 1:fieldcount(MPR2State)-2]...,0,:unknown)
end

function SolverCore.reset!(solver::MPR2Solver{T}) where {T}
  solver
end

function SolverCore.solve!(
  solver::MPR2Solver{T},
  MPnlp::FPMPNLPModel{H},
  stats::GenericExecutionStats;
  x₀::V = MPnlp.meta.x0,
  par::MPR2Params = MPR2Params(MPnlp.FPList[1],H),
  atol::H = H(sqrt(eps(MPnlp.FPList[end]))),
  rtol::H = H(sqrt(eps(MPnlp.FPList[end]))),
  max_eval::Int = -1,
  max_iter::Int = 1000,
  max_time::Float64 = 30.0,
  σmin::H = H(sqrt(MPnlp.FPList[end](MPnlp.EpsList[end]))),
  verbose::Int=0,
  e::E = nothing,
  compute_f_at_x! = compute_f_at_x_default!,
  compute_f_at_c! = compute_f_at_c_default!,
  compute_g! = compute_g_default!,
  recompute_g! = recompute_g_default!,
  selectPic! = selectPic_default!
) where {V<:AbstractVector{<:AbstractFloat}, H, T, E}

  unconstrained(MPnlp) || error("MPR2 should only be called on unconstrained problems.")
  SolverCore.reset!(stats)
  
  start_time = time()
  SolverCore.set_time!(stats, 0.0)

  # check for ill initialized parameters
  CheckMPR2ParamConditions(par)
  # instanciate MPR2 states
  state = MPR2State(H)
  # initialize parameters
  η₁ = par.η₁
  η₂ = par.η₂
  γ₁ = par.γ₁ # in lowest precision format to ensure type stability when updating σ
  γ₂ = par.γ₂ # in lowest precision format to ensure type stability when updating σ
  #initialize variables
  umpt!(solver.x, x₀)
  x = solver.x
  g = solver.g
  s = solver.s
  c = solver.c

  SolverCore.set_iter!(stats, 0)

  #initialize precisions
  πmax = length(MPnlp.FPList)
  π = MPR2Precisions(πmax)
  #misc initialization
  n = MPnlp.meta.nvar
  FP = MPnlp.FPList
  U = MPnlp.UList
  state.iter=0


  γfunc = MPnlp.γfunc
  βfunc(n::Int,u::H) = max(abs(1-sqrt(1-γfunc(n+2,u))),abs(1-sqrt(1+γfunc(n,u)))) # error bound on euclidean norm
  
  # initial evaluation, check for overflow
  compute_f_at_x!(MPnlp,state,π,par,e,x)
  if isinf(state.f) || isinf(state.ωf)
    @warn "Objective or ojective error overflow at x0"
    state.status = :exception
  end
  SolverCore.set_objective!(stats, state.f)

  compute_g!(MPnlp,state,π,par,e,x,g)
  if findfirst(x->x === FP[end](Inf), g) !== nothing || state.ωg == FP[end](Inf)
    @warn "Gradient or gradient error overflow at x0"
    state.status = :exception
  end
  umpt!(g,g[π.πg])
  state.g_norm = H(norm(g[π.πg]))
  SolverCore.set_dual_residual!(stats,state.g_norm)

  σ0 = 2^round(log2(state.g_norm+1)) # ensures ||s_0|| ≈ 1 with sigma_0 = 2^n with n an interger, i.e. sigma_0 is exactly representable in n bits 
  state.σ = H(σ0) # declare σ with the highest precision, convert it when computing sk to keep it at precision πg
  
  # check first order stopping condition
  ϵ = atol + rtol * state.g_norm
  optimal = state.g_norm ≤ (1-βfunc(n,U[π.πg]))*ϵ/(1+H(state.ωg))
  if optimal
    @info("First order critical point found at initial point")
  end
  if verbose > 0
    infoline = @sprintf "%6s  %9s  %9s  %9s  %9s  %9s  %7s  %7s  %7s  %7s %7s  %2s  %2s  %2s  %2s\n" "iter" "f(x)" "ωf(x)" "f(c)" "ωf(c)" "‖g‖" "ωg" "σ" "μ" "ϕ" "ρ" "πx" "πc" "πf" "πg"
    @info infoline
    infoline = @sprintf "%6d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %2d  %2d  %2d  %2d \n" stats.iter state.f state.ωf state.f⁺ state.ωf⁺ state.g_norm state.ωg state.σ state.μ state.ϕ state.ρ π.πx π.πc π.πf π.πg
    @info infoline
  end

  SolverCore.set_status!(
    stats,
    SolverCore.get_status(
      MPnlp,
      elapsed_time = stats.elapsed_time,
      optimal = optimal,
      max_eval = max_eval,
      iter = stats.iter,
      max_iter = max_iter,
      max_time = max_time,
    ),
  )
  if state.status == :exception
    set_status!(stats,:exception) # overflow case not covered by get_status()
  else
    state.status = stats.status
  end

  done = stats.status != :unknown

  #main loop
  while(!done)
    state.iter = stats.iter

    π.πs = π.πg
    computeStep!(s, g, state.σ, FP, π) || (state.status = :exception)
    computeCandidate!(c, x, s, FP, π) || (state.status = :exception)
    computeModelDecrease!(g, s, state, FP, π) || (state.status = :exception)
    
    g_recomp, prec_fail = recompute_g!(MPnlp,state,π,par,e,x,g,s)
    if prec_fail 
      state.status = :exception
    end
    if g_recomp #have to recompute everything depending on g
      umpt!(g,g[π.πg])
      computeStep!(s, g, state.σ, FP, π) || (state.status = :exception)
      computeCandidate!(c, x, s, FP, π) || (state.status = :exception)
      computeModelDecrease!(g, s, state, FP, π) || (state.status = :exception)
    end
    SolverCore.set_dual_residual!(stats,state.g_norm)
    
    prec_fail = compute_f_at_x!(MPnlp,state,π,par,e,x)
    if prec_fail
      state.status = :exception
    end
    SolverCore.set_objective!(stats, state.f)

    prec_fail = compute_f_at_c!(MPnlp,state,π,par,e,c)
    if prec_fail
      state.status = :exception
    end

    state.ρ = ( H(state.f) - H(state.f⁺)) / H(state.ΔT)
    
    if state.ρ ≥ η₂ 
      state.σ = max(σmin,  state.σ*γ₁)
    end
    if state.ρ < η₁
      state.σ = state.σ * γ₂
    end

    if state.ρ ≥ η₁
      prec_fail = compute_g!(MPnlp,state,π,par,e,c,g)
      if prec_fail 
        state.status = :exception
      end
      umpt!(g,g[π.πg])
      
      if findfirst(x->isinf(x), g[end]) !== nothing || isinf(state.ωg)
        @warn "gradient evaluation error at c too big to ensure convergence"
        state.status = :exception
      end
      
      state.g_norm = H(norm(g[π.πg]))
      umpt!(x,c[π.πc])
      state.f = state.f⁺
      SolverCore.set_objective!(stats,state.f⁺)
      state.ωf = state.ωf⁺
      
      π.πf = π.πf⁺
      π.πx = π.πc
      selectPic!(π)
    end

    SolverCore.set_iter!(stats, stats.iter + 1)
    state.iter = stats.iter
    SolverCore.set_time!(stats, time() - start_time)
    SolverCore.set_dual_residual!(stats, state.g_norm)
    optimal = state.g_norm ≤1/(1+βfunc(n,U[π.πg]))*ϵ/(1+H(state.ωg))

    if verbose > 0
      infoline = @sprintf "%6d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %2d  %2d  %2d  %2d \n" stats.iter state.f state.ωf state.f⁺ state.ωf⁺ state.g_norm state.ωg state.σ state.μ state.ϕ state.ρ π.πx π.πc π.πf π.πg
      @info infoline
    end

    SolverCore.set_status!(
      stats,
      SolverCore.get_status(
        MPnlp,
        elapsed_time = stats.elapsed_time,
        optimal = optimal,
        max_eval = max_eval,
        iter = stats.iter,
        max_iter = max_iter,
        max_time = max_time,
      ),
    )
    if state.status == :exception
      set_status!(stats,:exception) # overflow case (exception) not covered by get_status()
    else
      state.status = stats.status
    end

    done = stats.status != :unknown
  end

  SolverCore.set_solution!(stats, x[end])
  return stats
  
end

"""
    CheckUnderOverflowStep(s::AbstractVector)

Check if step over/underflow. Step cannot be zero in theory, if this happens it means that underflow occurs.

# Outpus:
* ::bool : true if over/underflow
"""

function CheckUnderOverflowStep(s::AbstractVector,g::AbstractVector)
  findall(x->x==0.0,s) != findall(x->x==0.0,g) || findfirst(x->isinf(x),s) !== nothing  # overflow or underflow occured, stop the loop
end

"""
    CheckUnderOverflowCandidate(c::AbstractVector)

Check if candidate over/underflow.

# Outputs: 
* ::bool : true if over/underflow occurs
"""

function CheckUnderOverflowCandidate(c::AbstractVector,x::AbstractVector)
  findfirst(x->x==Inf,c) !== nothing || c == x
end

"""
    CheckUnderOverflowMD(state::MPR2State)

Check if model decrease ΔT over/underflow

"""

function CheckUnderOverflowMD(ΔT::AbstractFloat)
  isinf(ΔT) || ΔT == 0.0  # overflow or underflow occured, stop the loop
end

"""
    umpt!(x::Tuple, y::Vector{S})

Update multi precision containers. 
Update is occuring only if precision input vector y is lower or equal to the one of the element of the container (vector) to avoid rounding error due to conversion. 
"""
function umpt!(x::Tuple, y::Vector{S}) where {S}
  for xel in x 
    eps(typeof(xel[1])) <= eps(S) && (xel .= y)
  end
end

####### Robust implementation of step, candidate and model decrease computation. #####
"""
    computeStep!(s::T, g::T, σ::H, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple, H}

Compute step with FP format avoiding underflow and overflow
# Arguments
* `s::Vector{T}` : step 
* `g::Vector{T}` : gradient 
* `πg::Int` : `g` FP index
* `σ::H` : regularization parameter
* `FP::Vector{Int}` : Available floating point formats

# Modified arguments :
* `s::Vector` : step vector container
* `π::MPR2Precisions` : hold the FP format indices
# Output
* ::bool : false if over/underflow occur, true otherwise
"""
function computeStep!(s::T, g::T, σ::H, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple, H}
  πmax = length(FP)
  while FP[π.πs](σ) === FP[π.πs](0) || isinf(FP[π.πs](σ)) # σ cast underflow or overflow
    if π.πs == πmax # over/underflow at max prec FP
      @error "over/underflow of σ with maximum precision FP format ($(FP[πmax]))"
      return false
    end
    π.πs += 1
  end
  s[π.πs] .= - g[π.πs]/FP[π.πs](σ)
  # increase FP prec until no overflow or underflow occurs
  while CheckUnderOverflowStep(s[π.πs],g[π.πg]) 
    if π.πs == πmax #not enough precision to avoid over/underflow 
      @error "over/underflow with maximum precision FP format ($(FP[πmax]))"
      return false
    end
    π.πs +=1 
    s[π.πs] .= - g[π.πs] / FP[π.πs](σ)
  end
  umpt!(s, s[π.πs])
  return true
end

"""
    computeCandidate!(c::T, x::T, s::T, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple}

Compute candidate with FP format avoiding underflow and overflow
# Arguments
* x::Vector{T} : incumbent 
* s::Vector{T} : step
* FP::Vector{Int} : Available floating point formats
* π::MPR2Precisions

# Modified arguments:
* c::Vector{T} : candidate
* π::MPR2Precisions : π.πc updated

# Outputs
* ::bool : false if over/underflow occur with highest precision FP format, true otherwise
"""
function computeCandidate!(c::T, x::T, s::T, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple}
  πmax = length(FP)
  πx = π.πx
  πs = π.πs
  πc = π.πc
  c[πc] .= FP[πc].(x[max(πc,πx)] .+ s[max(πc,πs)])
  while CheckUnderOverflowCandidate(c[πc],x[πx])
    if πc == πmax #not enough precision to avoid underflow
      @warn "over/underflow with maximum precision FP format ($(FP[πmax]))"
      π.πc = πc
      return false
    end
    πc +=1 
    c[πc] .= FP[πc].(x[max(πc,πx)] .+ s[max(πc,πs)])
  end
  umpt!(c, c[πc])
  π.πc = πc
  return true
end


"""
    computeModelDecrease!(g::T,s::T,st::MPR2State{H},FP::Vector{DataType},π::MPR2Precisions) where {T <: Tuple}

Compute model decrease with FP format avoiding underflow and overflow

# Arguments
* g::Vector{T} : gradient 
* s::Vector{T} : step
* st::MPR2State{H}: algo status
* FP::Vector{Int} : Available floating point formats
* π::MPR2Precisions : hold the FP format indices

# Modified Arguments
* st::MPR2State{H} : st.ΔT updated 
* π::MPR2Precisions : π.πΔ updated 

# Outputs
* ::bool : false if over/underflow occur with highest precision FP format, true otherwise

"""
function computeModelDecrease!(g::T,s::T,st::MPR2State{H},FP::Vector{DataType},π::MPR2Precisions) where {H, T <: Tuple}
  πΔ = π.πΔ
  πmax = length(FP)
  if πΔ < max(π.πs,π.πg)
    error("Model decrease computation FP format should be greater that FP format of g and s")
  end
  st.ΔT = H.(- dot(g[πΔ], s[πΔ])) 
  while CheckUnderOverflowMD(st.ΔT)
    if πΔ == πmax
      @warn "over/underflow with maximum precision FP format ($(FP[πmax]))"
      π.πΔ = πΔ
      return false
    end
    πΔ += 1
    st.ΔT = H.(- dot(g[πΔ], s[πΔ])) 
  end
  π.πΔ = πΔ
  return true
end

####### Default strategy for precision selections #######
"""
    computeMu(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions) where H

Compute μ value for gradient error ωg, ratio ϕ = ||x||/||s|| and rounding error models
"""
function computeMu(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions) where H
  n = m.meta.nvar
  αfunc(n::Int,u::H) = 1/(1-m.γfunc(n,u))
  u = π.πc >= π.πs ? m.UList[π.πc] : m.UList[π.πc] + m.UList[π.πs] + m.UList[π.πc]*m.UList[π.πs]
  return ( αfunc(n+1,m.UList[π.πΔ]) * H(st.ωg) * (m.UList[π.πc] + st.ϕ * u +1) + αfunc(n+1,m.UList[π.πΔ]) * u * (st.ϕ +1) + m.UList[π.πg] + m.γfunc(n+2,m.UList[π.πΔ]) * αfunc(n+1,m.UList[π.πΔ]) ) / (1-m.UList[π.πs])
end

"""
    recomputeMuPrecSelection!(π::MPR2Precisions, πr::MPR2Precisions, πmax::Int64)

Default strategy to select new precisions to recompute μ in the case where μ > κₘ
Returns a MPR2Precisions containing new precisions with which to recompute mu.
"""
function recomputeMuPrecSelection!(π::MPR2Precisions, πr::MPR2Precisions, πmax::Int64)
  minprec = min(π.πnx,π.πns)
  # Priority #1 strategy: increase πΔ to decrease model decrease error
  if π.πΔ < πmax
    πr.πΔ = π.πΔ + 1
  # Priority #2 strategy: increase norm computation precision for x and s to decrease ϕ
  elseif minprec < πmax
    πr.πnx = max(π.πnx,minprec+1)
    πr.πns = max(π.πns,minprec+1)
  # Priority #3 strategy: increase πc
  elseif π.πs < πmax
    πr.πs = π.πs + 1
  elseif π.πc < πmax
    πr.πc = π.πc + 1
  # Priority #4 strategy: recompute gradient
  elseif π.πg < πmax
    πr.πg = π.πg + 1
  else
    πr.πx = -1 # max precision reached code
  end
end

""" 
    recomputeMu!(m::FPMPNLPModel{H}, x::T, g::T, s::T, st::MPR2State{H}, π::MPR2Precisions, πr::MPR2Precisions) where {T <: Tuple, H}

Recompute mu based on new precision levels. 
Performs only necessary steps of solve! main loop to recompute mu. 
Possible step to recompute are:
- recompute ϕhat and ϕ with higher FP format for norm computation of x and s
- recompute step with greater precision: decrease μ denominator
- recompute gradient with higher precision to decrease ωg
- recompute candidate with higher prec FP format to decrease u
- recompute model reduction with higher precision to decrease αfunc(n,U[π.πΔ])
Does not make the over/underflow check as in main loop, since it is a repetition of the main loop with higher precisions and these issue shouldn't occur
# Outputs:
* b : true if gradient has been modified, false otherwise
See [`recomputeMuPrecSelection!`](@ref)
"""
function recomputeMu!(m::FPMPNLPModel{H}, x::T, g::T, s::T, st::MPR2State{H}, π::MPR2Precisions, πr::MPR2Precisions) where {T <: Tuple, H}
  g_recompute = false
  n = m.meta.nvar
  βfunc(n::Int,u::H) = max(abs(1-sqrt(1-m.γfunc(n+2,u))),abs(1-sqrt(1+m.γfunc(n,u)))) # error bound on euclidean norm
  if π.πΔ != πr.πΔ # recompute model decrease
    st.ΔT = computeModelDecrease!(g, s, st, m.FPList, πr)
  end
  if π.πnx != πr.πnx || π.πns != πr.πns # recompute x, s norm and ϕ
    if π.πnx != πr.πnx
      st.x_norm = H(norm(x[πr.πnx]))
    end
    if π.πns != πr.πns
      st.s_norm = H(norm(s[πr.πns]))
    end
    st.ϕhat = st.x_norm/st.s_norm
    st.ϕ = st.ϕhat * (1+βfunc(n,m.UList[πr.πnx]))/(1-βfunc(n,m.UList[πr.πns]))
  end
  if π.πs != πr.πs
    computeStep!(s,g,st.σ,m.FPList,πr)
  end
  if π.πg != πr.πg
    st.ωg = graderrmp!(m, x[πr.πg], g[πr.πg])
    g_recompute = true
    umpt!(g,g[π.πg])
  end
  st.μ = computeMu(m, st, πr)
  return g_recompute
end
  
""" 
    selectPif!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, ωfBound::H) where H

Select a precision for objective evaluation for candidate based on predicted evaluation error.
Evaluation is predicted as:
* Relative error:
  + Predicted value of objective at c: f(c) ≈ f(x) - ΔTk
  + Relative error model: ωf(c) = |f(c)| * RelErr
* Interval error:
  + Predicted value of objective at c: f(c) ≈ f(x) - ΔTk
  + Interval evaluation error is proportional to f(x)
  + Interval evaluation error depends linearly with unit-roundoff 
* Other: Lowest precision that does not cast candidate in a lower prec FP format and f(c) predicted does not overflow
"""
function selectPif!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, ωfBound::H) where H
  πmax = length(m.FPList)
  πmin_no_ov = findfirst(x -> x > abs(st.f) - st.ΔT, m.OFList) # lowest precision level such that predicted f(ck) ≈ fk+gk'ck does not overflow
  if πmin_no_ov === nothing
    π.πf⁺ = πmax
    return π
  end
  πmin = max(π.πc,πmin_no_ov) # lower bound on πf⁺ to avoid casting error on c and possible overflow
  f⁺_pred = st.f-st.ΔT 
  ωf⁺_pred = 0
  if m.ObjEvalMode == REL_ERR
    ωf⁺_pred = f⁺_pred .* m.ωfRelErr
  elseif m.ObjEvalMode == INT_ERR 
    r = abs(f⁺_pred)/abs(st.f)
    ωf⁺_pred = st.ωf * r * m.UList ./ m.UList[π.πf]
  else 
    π.πf⁺ = πmin
    return π
  end
  πf⁺_pred = findfirst(x -> x<ωfBound, ωf⁺_pred)
  π.πf⁺ = πf⁺_pred === nothing ? πmax : max(πf⁺_pred,πmin)
  return π
end

function selectPic_default!(π::MPR2Precisions)
  π.πc = max(1,π.πf⁺-1)
end

####### Defautl callback function for objective and gradient evaluation #########

function compute_f_at_c_default!(m::FPMPNLPModel{H}, st::MPR2State{H}, π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T) where {H, L, E, T <: Tuple}
  prec_fail = false
  ωfBound = p.η₀*st.ΔT
  selectPif!(m, st, π, ωfBound) # select precision evaluation 
  st.f⁺, st.ωf⁺, π.πf⁺ = objReachPrec(m, c, ωfBound, π = π.πf⁺)
  if isinf(st.f⁺) || isinf(st.ωf⁺)
    @warn "Objective evaluation or error at c overflow"
    prec_fail = true
  end
  if st.ωf⁺ > ωfBound
    @warn "Objective evaluation error at c too big"
    prec_fail = true
  end
  return prec_fail
end

function compute_f_at_x_default!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, x::T) where {H, L, E, T <: Tuple}
  prec_fail = false
  πmax = length(m.FPList)
  if st.iter == 0
    st.f, st.ωf, π.πf = objReachPrec(m, x, m.OFList[end], π = π.πf)
  else
    ωfBound = p.η₀*st.ΔT
    if st.ωf > ωfBound
      if π.πf == πmax
        @warn "Objective evaluation error at x too big to ensure convergence"
        st.status = :exception
        prec_fail = true
        return prec_fail
      end
      π.πf += 1 # default strategy
      st.f, st.ωf, π.πf = objReachPrec(m, x, ωfBound, π = π.πf)
      if isinf(st.f) ||  isinf(st.ωf)
        @warn "Objective evaluation or error overflow at x"
        st.status = :exception
        prec_fail = true
        return prec_fail
      end
      if st.ωf > ωfBound
        @warn "Objective evaluation error at x too big to ensure convergence"
        st.status = :exception
        prec_fail = true
        return prec_fail
      end
    end
  end
  return prec_fail
end

function compute_g_default!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, c::T, g::T) where {H, L, E, T <: Tuple}
  π.πg = π.πc # default strategy, could be a callback
  st.ωg, π.πg = gradReachPrec!(m, c, g, m.OFList[end], π = π.πg)
  return false
end

function recompute_g_default!(m::FPMPNLPModel{H}, st::MPR2State{H},  π::MPR2Precisions, p::MPR2Params{H, L}, e::E, x::T, g::T, s::T) where {H, L, E, T <: Tuple}
  g_recompute = false
  prec_fail = false
  πmax = length(m.FPList)
  n = m.meta.nvar
  βfunc(n::Int,u::H) = max(abs(1-sqrt(1-m.γfunc(n+2,u))),abs(1-sqrt(1+m.γfunc(n,u)))) # error bound on euclidean norm
  ##### default choice, can be put in a callback for alternative strategy ####
  π.πnx = min(π.πx+1,πmax) # use better prec to lower βn and therefore phik therefore muk
  π.πns = min(π.πs+1,πmax) 
  ##################################################
  st.x_norm = H(norm(x[π.πnx]))
  st.s_norm = H(norm(s[π.πns]))
  st.ϕhat = st.x_norm/st.s_norm
  st.ϕ = st.ϕhat * (1+βfunc(n,m.UList[π.πnx]))/(1-βfunc(n,m.UList[π.πns]))
  prec = findall(u->u<1/st.ϕ,m.UList) 
  if isempty(prec) # step size too small compared to incumbent
    @warn "Algo stops because the step size is too small compare to the incumbent, addition unstable (due to rounding error or absorbtion) with highest precision level"
    st.status = :small_step
    return false, false
  end
  st.μ = computeMu(m, st, π)
  while st.μ > p.κₘ && !prec_fail
    πr = copy(π)
    recomputeMuPrecSelection!(π, πr, πmax)
    if st.μ > p.κₘ && πr.πx == -1
      @warn "Gradient error too big with maximum precision FP format: μ ($(st.μ)) > κₘ($(p.κₘ))"
      prec_fail = true
      return g_recompute, prec_fail
    end
    g_recompute = recomputeMu!(m, x, g, s, st, π, πr)
    update_struct!(π,πr)
  end
  return g_recompute, prec_fail
end

end # module
