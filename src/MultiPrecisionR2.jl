# Author: D. Monnet

# Multi precision regularization algorithm.

# This work has been supported by the NSERC Alliance grant 544900- 19 in collaboration with Huawei-Canada. 

module MultiPrecisionR2

using ADNLPModels, IntervalArithmetic, NLPModels, Printf, LinearAlgebra, SolverCore

export MPR2, MPR2Solver, MPR2Params, MPR2Precisions, solve!, umpt!, update_struct!

abstract type AbstractMPNLPModel{T, S} <: AbstractNLPModel{T, S} end

include("MPCounters.jl")
include("MPNLPModels.jl")
include("utils.jl")

"""
    MPR2Precisions(π::Int)

FP format index storage structure.

Stores FP formats index in `FPMPNLPModel.FPList` of obj, grad, model reduction and norms evaluations, and FP format index of MPR2 algorithm vector variables. 

# Fields
- `πx::Int` : Current FP format of current incumbent `x`
- `πnx::Int` : FP format used for `x` norm evaluation
- `πs::Int` : Current FP format of step`s`
- `πns::Int` : FP format used for `s` norm evaluation
- `πc::Int` : Current FP format of candidate `c`
- `πf::Int` : FP format used for objective evaluation at `x`
- `πf⁺::Int` : FP format used for objective evaluation at `c`
- `πg::Int` : FP format used for gradient evaluation at `c`
- `πΔ::Int` : FP format used for model reduction computation
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

Base.copy(π::MPR2Precisions) =
  MPR2Precisions(π.πx, π.πnx, π.πs, π.πns, π.πc, π.πf, π.πf⁺, π.πg, π.πΔ)

"""
    MPR2Params(LPFormat::DataType, HPFormat::DataType)
    
MPR2 parameters structure.

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
function CheckMPR2ParamConditions(p::MPR2Params)
  0 ≤ p.η₀ ≤ 1 / 2 * p.η₁ || error("Expected 0 ≤ η₀ ≤ 1/2*η₁")
  p.η₁ ≤ p.η₂ < 1 || error("Expected η₁ ≤ η₂ < 1")
  p.η₀ + p.κₘ / 2 ≤ 0.5 * (1 - p.η₂) || error("Expected η₀+κₘ/2 ≤0.5*(1-η₂)")
  0 < p.γ₁ < 1 < p.γ₂ || error("Expected 0<γ₁<1<γ₂")
end

"""
    MPR2Solver(MPnlp::FPMPNLPModel)

Solver structure containing all the variables necessary to MRP2.

# Fields:

- `x::T`: incumbent
- g::T : gradient
- s::T : step 
- c::T : candidate
- π::MPR2Precisions : FP format indices (precision) structure
- p::MPR2Params : MPR2 parameters
- x_norm::H : norm of `x`
- s_norm::H : norm of `s`
- g_norm::H : norm of `g`
- ΔT::H : model decrease
- ρ::H : success ratio
- ϕ::H : guaranteed upper bound on ||x||/||s||
- ϕhat::H : computed value of ||x||/||s||
- μ::H : error indicator
- f::H : objective value at `x`
- f⁺::H : objective value at `c`
- ωf::H : objective evaluation error at `x`, |f(x) - fl(f(x))| <= `ωf`
- ωf⁺::H : objective evaluation error at `c`, |f(c) - fl(f(c))| <= `ωf⁺`
- ωg::H : gradient evaluation error at `c`, ||∇f(c) - fl(∇f(c))||₂ <= `ωg`||fl(∇f(c))||₂
- ωfBound::H : error tolerance on objective evaluation
- σ::H : regularization parameter
- πmax::Int : number of FP formats available for evaluations
- init::Bool : initialized with `true`, set to `false` when entering main loop 
"""
mutable struct MPR2Solver{T <: Tuple, H <: AbstractFloat} <: AbstractOptimizationSolver
  x::T
  g::T
  s::T
  c::T
  π::MPR2Precisions
  p::MPR2Params
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
  πmax::Int
  init::Bool
end

function MPR2Solver(MPnlp::M) where {S, H, B, D, M <: FPMPNLPModel{H, B, D, S}}
  nvar = MPnlp.meta.nvar
  x = Tuple(Vector{ElType}(undef, nvar) for ElType in MPnlp.FPList)
  s = Tuple(Vector{ElType}(undef, nvar) for ElType in MPnlp.FPList)
  c = Tuple(Vector{ElType}(undef, nvar) for ElType in MPnlp.FPList)
  g = Tuple(Vector{ElType}(undef, nvar) for ElType in MPnlp.FPList)
  πmax = length(MPnlp.FPList)
  π = MPR2Precisions(πmax)
  par = MPR2Params(MPnlp.FPList[1], H)
  return MPR2Solver(
    x,
    g,
    s,
    c,
    π,
    par,
    [H(0) for _ = 1:(fieldcount(MPR2Solver) - 8)]...,
    πmax,
    true,
  )
end


"""
    MPR2(MPnlp; kwargs...)
An implementation of the quadratic regularization algorithm with dynamic selection of floating point format for objective and gradient evaluation, robust against finite precision rounding errors.
Type parameters are: `S::AbstractVector`, `H::AbstractFloat`, `T::AbstractFloat`, `E::DataType` 

# Arguments
- `MPnlp::FPMPNLPModel` : Multi precision model, see `FPMPNLPModel`

Keyword agruments:
- `x₀::S = MPnlp.Model.meta.x0` : initial guess 
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
1. `GenericExecutionStats`: execution stats containing information about algorithm execution (nb. of iteration, termination status, ...). See `SolverCore.jl`

# Example
```julia
T = [Float32, Float64]
f(x) = sum(x.^2)
x = ones(Float32,2)
mpnlp = FPMPNLPModel(f,x,T)
MPR2(mpnlp)
```
"""
function MPR2(MPnlp::FPMPNLPModel; kwargs...)
  solver = MPR2Solver(MPnlp)
  return SolverCore.solve!(solver, MPnlp; kwargs...)
end

"""
    update_struct!(str,other_str)

Update the fields of `str` with the fields of `other_str`.
"""
function update_struct!(str::MPR2Precisions, other_str::MPR2Precisions)
  for f in fieldnames(MPR2Precisions)
    setfield!(str, f, getfield(other_str, f))
  end
end

function SolverCore.reset!(solver::MPR2Solver{T}) where {T}
  solver
end

function SolverCore.solve!(
  solver::MPR2Solver{T, H},
  MPnlp::FPMPNLPModel,
  stats::GenericExecutionStats;
  x₀ = MPnlp.meta.x0,
  par::MPR2Params = MPR2Params(MPnlp.FPList[1], H),
  atol::H = H(sqrt(eps(MPnlp.FPList[end]))),
  rtol::H = H(sqrt(eps(MPnlp.FPList[end]))),
  max_eval::Int = -1,
  max_iter::Int = 1000,
  max_time::Float64 = 30.0,
  σmin::H = H(sqrt(MPnlp.FPList[end](MPnlp.EpsList[end]))),
  verbose::Int = 0,
  e::E = nothing,
  compute_f_at_x! = compute_f_at_x_default!,
  compute_f_at_c! = compute_f_at_c_default!,
  compute_g! = compute_g_default!,
  recompute_g! = recompute_g_default!,
  selectPic! = selectPic_default!,
) where {H, T, E}
  unconstrained(MPnlp) || error("MPR2 should only be called on unconstrained problems.")
  SolverCore.reset!(stats)

  start_time = time()
  SolverCore.set_time!(stats, 0.0)

  # check for ill initialized parameters
  CheckMPR2ParamConditions(par)
  solver.p = par
  # initialize parameters
  η₁ = par.η₁
  η₂ = par.η₂
  γ₁ = par.γ₁ # in lowest precision format to ensure type stability when updating σ
  γ₂ = par.γ₂ # in lowest precision format to ensure type stability when updating σ

  #initialize init sol
  umpt!(solver.x, x₀)
  umpt!(solver.c, x₀)

  #misc initialization
  SolverCore.set_iter!(stats, 0)
  n = MPnlp.meta.nvar
  FP = MPnlp.FPList
  U = MPnlp.UList

  γfunc = MPnlp.γfunc
  βfunc(n::Int, u::H) = max(abs(1 - sqrt(1 - γfunc(n + 2, u))), abs(1 - sqrt(1 + γfunc(n, u)))) # error bound on euclidean norm

  # initial evaluation, check for overflow
  compute_f_at_x!(MPnlp, solver, stats, e)
  if isinf(solver.f) || isinf(solver.ωf)
    @warn "Objective or ojective error overflow at x0"
    stats.status = :exception
  end

  stats.objective = solver.f
  #SolverCore.set_objective!(stats, solver.f)

  compute_g!(MPnlp, solver, stats, e)
  if findfirst(x -> x === FP[end](Inf), solver.g) !== nothing || isinf(solver.ωg)
    @warn "Gradient or gradient error overflow at x0"
    stats.status = :exception
  end
  umpt!(solver.g, solver.g[solver.π.πg])
  solver.g_norm = H(norm(solver.g[solver.π.πg]))
  
  stats.dual_feas = solver.g_norm
  #SolverCore.set_dual_residual!(stats, solver.g_norm)

  σ0 = 2^round(log2(solver.g_norm + 1)) # ensures ||s_0|| ≈ 1 with sigma_0 = 2^n with n an interger, i.e. sigma_0 is exactly representable in n bits 
  solver.σ = H(σ0) # declare σ with the highest precision, convert it when computing sk to keep it at precision πg

  # check first order stopping condition
  ϵ = atol + rtol * solver.g_norm
  optimal = solver.g_norm ≤ (1 - βfunc(n, U[solver.π.πg])) * ϵ / (1 + H(solver.ωg))
  if optimal
    @info("First order critical point found at initial point")
  end
  if verbose > 0
    infoline =
      @sprintf "%6s  %9s  %9s  %9s  %9s  %9s  %7s  %7s  %7s  %7s %7s  %2s  %2s  %2s  %2s\n" "iter" "f(x)" "ωf(x)" "f(c)" "ωf(c)" "‖g‖" "ωg" "σ" "μ" "ϕ" "ρ" "πx" "πc" "πf" "πg"
    @info infoline
    infoline =
      @sprintf "%6d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %2d  %2d  %2d  %2d \n" stats.iter solver.f solver.ωf solver.f⁺ solver.ωf⁺ solver.g_norm solver.ωg solver.σ solver.μ solver.ϕ solver.ρ solver.π.πx solver.π.πc solver.π.πf solver.π.πg
    @info infoline
  end

  if stats.status != :exception # get_status does not handle overflow case that set status to exception 
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
  end

  done = stats.status != :unknown
  solver.init = false
  #main loop
  while (!done)
    solver.π.πs = solver.π.πg
    computeStep!(solver.s, solver.g, solver.σ, FP, solver.π) || (stats.status = :exception)
    computeCandidate!(solver.c, solver.x, solver.s, FP, solver.π) || (stats.status = :exception)
    computeModelDecrease!(solver.g, solver.s, solver, FP, solver.π) || (stats.status = :exception)

    g_recomp, g_succ = recompute_g!(MPnlp, solver, stats, e)
    if !g_succ && stats.status != :small_step
      stats.status = :exception
    end
    if g_recomp #have to recompute everything depending on g
      umpt!(solver.g, solver.g[solver.π.πg])
      computeStep!(solver.s, solver.g, solver.σ, FP, solver.π) || (stats.status = :exception)
      computeCandidate!(solver.c, solver.x, solver.s, FP, solver.π) || (stats.status = :exception)
      computeModelDecrease!(solver.g, solver.s, solver, FP, solver.π) || (stats.status = :exception)
    end
    stats.dual_feas = solver.g_norm
    #SolverCore.set_dual_residual!(stats, solver.g_norm)

    compute_f_at_x!(MPnlp, solver, stats, e) || (stats.status = :exception)
    stats.objective = solver.f
    #SolverCore.set_objective!(stats, solver.f)

    compute_f_at_c!(MPnlp, solver, stats, e) || (stats.status = :exception)

    solver.ρ = (H(solver.f) - H(solver.f⁺)) / H(solver.ΔT)

    if solver.ρ ≥ η₂
      solver.σ = max(σmin, solver.σ * γ₁)
    end
    if solver.ρ < η₁
      solver.σ = solver.σ * γ₂
    end

    if solver.ρ ≥ η₁
      compute_g!(MPnlp, solver, stats, e) || (stats.status = :exception)
      umpt!(solver.g, solver.g[solver.π.πg])

      if findfirst(x -> isinf(x), solver.g[end]) !== nothing || isinf(solver.ωg)
        @warn "gradient evaluation error at c too big to ensure convergence"
        stats.status = :exception
      end

      solver.g_norm = H(norm(solver.g[solver.π.πg]))
      umpt!(solver.x, solver.c[solver.π.πc])
      solver.f = solver.f⁺
      stats.objective = solver.f⁺
      #SolverCore.set_objective!(stats, solver.f⁺)
      solver.ωf = solver.ωf⁺

      solver.π.πf = solver.π.πf⁺
      solver.π.πx = solver.π.πc
      selectPic!(solver)
    end

    SolverCore.set_iter!(stats, stats.iter + 1)
    SolverCore.set_time!(stats, time() - start_time)
    stats.dual_feas = solver.g_norm
    #SolverCore.set_dual_residual!(stats, solver.g_norm)
    optimal = solver.g_norm ≤ 1 / (1 + βfunc(n, U[solver.π.πg])) * ϵ / (1 + H(solver.ωg))

    if verbose > 0
      infoline =
        @sprintf "%6d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %2d  %2d  %2d  %2d \n" stats.iter solver.f solver.ωf solver.f⁺ solver.ωf⁺ solver.g_norm solver.ωg solver.σ solver.μ solver.ϕ solver.ρ solver.π.πx solver.π.πc solver.π.πf solver.π.πg
      @info infoline
    end

    if stats.status != :exception # get_status does not handle overflow case that set status to exception 
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
    end

    done = stats.status != :unknown
  end

  stats.solution = solver.x[end]
  #SolverCore.set_solution!(stats, solver.x[end])
  return stats
end

"""
    CheckUnderOverflowStep(s::AbstractVector)

Check if step `s` over/underflow. Step cannot be zero in theory, if this happens it means that underflow occurs.

# Outputs:
1. ::bool : true if over/underflow
"""

function CheckUnderOverflowStep(s::AbstractVector, g::AbstractVector)
  of = findfirst(x -> isinf(x), s) !== nothing
  uf = findall(x -> x == 0.0, s) != findall(x -> x == 0.0, g)
  return of || uf
end

"""
    CheckUnderOverflowCandidate(c::AbstractVector)

Check if candidate `c` over/underflow.

# Outputs: 
1. ::bool : true if over/underflow occurs
"""

function CheckUnderOverflowCandidate(c::AbstractVector, x::AbstractVector, s::AbstractVector)
  of = findfirst(x -> isinf(x), c) !== nothing
  uf = (x .== c) != (s .== 0)
  return of || uf
end

"""
    CheckUnderOverflowMD(ΔT)

Check if model decrease ΔT over/underflow. ΔT==0 implies underflow since it is theoretically not possible to have this value (obtainable only for null gradient but MPR2 would have terminated).

# Outputs
1. ::bool : true if over/underflow occurs
"""

function CheckUnderOverflowMD(ΔT::AbstractFloat)
  isinf(ΔT) || ΔT == 0.0  # overflow or underflow occured, stop the loop
end

"""
    umpt!(x::Tuple, y::Vector{S})

Update the elements of the multi precision containers `x` with the value `y`.
Only the elements of `x` of FP formats with precision greater or equal than `y` are updated (avoid rounding error).

"""
function umpt!(x::Tuple, y::Vector{S}) where {S}
  for xel in x
    eps(typeof(xel[1])) <= eps(S) && (xel .= y)
  end
end

####### Robust implementation of step, candidate and model decrease computation. #####
"""
    computeStep!(s::T, g::T, σ::H, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple, H}

Compute step with proper FP format to avoid underflow and overflow

# Arguments
* `s::T` : step 
* `g::T` : gradient 
* `πg::Int` : `g` FP index
* `σ::H` : regularization parameter
* `FP::Vector{Int}` : Available floating point formats
* `π::MPR2Precisions` : FP format index storage structure 

# Modified arguments :
* `s::T` : updated with computed step
* `π::MPR2Precisions` : `π.πs` updated with FP format used for step computation

# Outputs
* `::bool` : false if over/underflow occurs, true otherwise
"""
function computeStep!(
  s::T,
  g::T,
  σ::H,
  FP::Vector{DataType},
  π::MPR2Precisions,
) where {T <: Tuple, H}
  πmax = length(FP)
  while FP[π.πs](σ) === FP[π.πs](0) || isinf(FP[π.πs](σ)) # σ cast underflow or overflow
    if π.πs == πmax # over/underflow at max prec FP
      @error "over/underflow of σ with maximum precision FP format ($(FP[πmax]))"
      return false
    end
    π.πs += 1
  end
  s[π.πs] .= -g[π.πs] / FP[π.πs](σ)
  # increase FP prec until no overflow or underflow occurs
  while CheckUnderOverflowStep(s[π.πs], g[π.πg])
    if π.πs == πmax #not enough precision to avoid over/underflow 
      @error "over/underflow with maximum precision FP format ($(FP[πmax]))"
      return false
    end
    π.πs += 1
    s[π.πs] .= -g[π.πs] / FP[π.πs](σ)
  end
  umpt!(s, s[π.πs])
  return true
end

"""
    computeCandidate!(c::T, x::T, s::T, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple}

Compute candidate with proper FP format to avoid underflow and overflow
# Arguments
* `x::T` : incumbent 
* `s::T` : step
* `FP::Vector{Int}` : Available floating point formats
* `π::MPR2Precisions` : FP format index storage structure

# Modified arguments:
* `c::T` : updated with candidate
* `π::MPR2Precisions` : `π.πc` updated with FP format index used for computation
 
# Outputs:
* `::bool` : false if over/underflow occur with highest precision FP format, true otherwise
"""
function computeCandidate!(
  c::T,
  x::T,
  s::T,
  FP::Vector{DataType},
  π::MPR2Precisions,
) where {T <: Tuple}
  πmax = length(FP)
  πx = π.πx
  πs = π.πs
  πc = π.πc
  c[πc] .= FP[πc].(x[max(πc, πx)] .+ s[max(πc, πs)])
  while CheckUnderOverflowCandidate(c[πc], x[πx], s[πs])
    if πc == πmax #not enough precision to avoid underflow
      @warn "over/underflow with maximum precision FP format ($(FP[πmax]))"
      π.πc = πc
      return false
    end
    πc += 1
    c[πc] .= FP[πc].(x[max(πc, πx)] .+ s[max(πc, πs)])
  end
  umpt!(c, c[πc])
  π.πc = πc
  return true
end

"""
    computeModelDecrease!(g::T,s::T,solver::MPR2Solver,FP::Vector{DataType},π::MPR2Precisions) where {T <: Tuple}

Compute model decrease with FP format avoiding underflow and overflow

# Arguments
* `g::T` : gradient 
* `s::T` : step
* `solver::MPR2Solver` : solver structure, stores intermediate variables
* `FP::Vector{Int}` : Available floating point formats
* `π::MPR2Precisions` : FP format index storage structure

# Modified Arguments
* `solver::MPR2Solver` : `solver.ΔT` updated 
* `π::MPR2Precisions` : `π.πΔ` updated 

# Outputs
* ::bool : false if over/underflow occur with highest precision FP format, true otherwise

"""
function computeModelDecrease!(
  g::T,
  s::T,
  solver::MPR2Solver{T, H},
  FP::Vector{DataType},
  π::MPR2Precisions,
) where {H, T <: Tuple}
  πΔ = π.πΔ
  πmax = length(FP)
  if πΔ < max(π.πs, π.πg)
    error("Model decrease computation FP format should be greater that FP format of g and s")
  end
  solver.ΔT = H.(-dot(g[πΔ], s[πΔ]))
  while CheckUnderOverflowMD(solver.ΔT)
    if πΔ == πmax
      @warn "over/underflow with maximum precision FP format ($(FP[πmax]))"
      π.πΔ = πΔ
      return false
    end
    πΔ += 1
    solver.ΔT = H.(-dot(g[πΔ], s[πΔ]))
  end
  π.πΔ = πΔ
  return true
end

"""
    computeMu(m::FPMPNLPModel, solver::MPR2Solver{T,H}; π::MPR2Precisions = solver.π)

Compute μ value for gradient error ωg, ratio ϕ = ||x||/||s|| and rounding error models
"""
function computeMu(m::FPMPNLPModel, solver::MPR2Solver{T, H}; π = solver.π) where {T, H}
  n = m.meta.nvar
  αfunc(n::Int, u::H) = 1 / (1 - m.γfunc(n, u))
  u = π.πc >= π.πs ? m.UList[π.πc] : m.UList[π.πc] + m.UList[π.πs] + m.UList[π.πc] * m.UList[π.πs]
  return (
    αfunc(n + 1, m.UList[π.πΔ]) * H(solver.ωg) * (m.UList[π.πc] + solver.ϕ * u + 1) +
    αfunc(n + 1, m.UList[π.πΔ]) * u * (solver.ϕ + 1) +
    m.UList[π.πg] +
    m.γfunc(n + 2, m.UList[π.πΔ]) * αfunc(n + 1, m.UList[π.πΔ])
  ) / (1 - m.UList[π.πs])
end

"""
    recomputeMuPrecSelection!(π::MPR2Precisions, πr::MPR2Precisions, πmax)

Default strategy to select new precisions to recompute μ in the case where μ > κₘ. Return false if no precision can be increased.
# Modified arguments:
* `πr`: contains new precision that will be used to recompute mu, see [`recomputeMu!`](@ref)
# Ouptputs
* `max_prec::bool` : return true if maximum precision levels have been reached
"""
function recomputeMuPrecSelection!(π::MPR2Precisions, πr::MPR2Precisions, πmax)
  minprec = min(π.πnx, π.πns)
  # Priority #1 strategy: increase πΔ to decrease model decrease error
  if π.πΔ < πmax
    πr.πΔ = π.πΔ + 1
    # Priority #2 strategy: increase norm computation precision for x and s to decrease ϕ
  elseif minprec < πmax
    πr.πnx = max(π.πnx, minprec + 1)
    πr.πns = max(π.πns, minprec + 1)
    # Priority #3 strategy: increase πc
  elseif π.πs < πmax
    πr.πs = π.πs + 1
  elseif π.πc < πmax
    πr.πc = π.πc + 1
    # Priority #4 strategy: recompute gradient
  elseif π.πg < πmax
    πr.πg = π.πg + 1
  else
    return true
  end
  return false
end

""" 
    recomputeMu!(m::FPMPNLPModel, solver::MPR2Solver{T,H}, πr::MPR2Precisions) where {T <: Tuple, H}

Recompute mu based on new precision levels. 
Performs only necessary operations to recompute mu. 
Possible operations are:
- recompute candidate with higher prec FP format to decrease u
- recompute ϕhat and ϕ with higher FP format for norm computation of x and s
- recompute step with greater precision: decrease μ denominator
- recompute gradient with higher precision to decrease ωg
- recompute model reduction with higher precision to decrease αfunc(n,U[π.πΔ])
Does not make the over/underflow check as in main loop, since it is a repetition of the main loop with higher precisions and these issue shouldn't occur
# Outputs:
* `g_recompute::Bool` : true if gradient has been modified, false otherwise
See [`recomputeMuPrecSelection!`](@ref)
"""
function recomputeMu!(
  m::FPMPNLPModel,
  solver::MPR2Solver{T, H},
  πr::MPR2Precisions,
) where {T <: Tuple, H}
  g_recompute = false
  n = m.meta.nvar
  βfunc(n::Int, u::H) = max(abs(1 - sqrt(1 - m.γfunc(n + 2, u))), abs(1 - sqrt(1 + m.γfunc(n, u)))) # error bound on euclidean norm
  if solver.π.πΔ != πr.πΔ # recompute model decrease
    solver.ΔT = computeModelDecrease!(solver.g, solver.s, solver, m.FPList, πr)
  end
  if solver.π.πnx != πr.πnx || solver.π.πns != πr.πns # recompute x, s norm and ϕ
    if solver.π.πnx != πr.πnx
      solver.x_norm = H(norm(solver.x[πr.πnx]))
    end
    if solver.π.πns != πr.πns
      solver.s_norm = H(norm(solver.s[πr.πns]))
    end
    solver.ϕhat = solver.x_norm / solver.s_norm
    solver.ϕ = solver.ϕhat * (1 + βfunc(n, m.UList[πr.πnx])) / (1 - βfunc(n, m.UList[πr.πns]))
  end
  if solver.π.πs != πr.πs
    computeStep!(solver.s, solver.g, solver.σ, m.FPList, πr)
  end
  if solver.π.πg != πr.πg
    solver.ωg = graderrmp!(m, solver.x[πr.πg], solver.g[πr.πg])
    g_recompute = true
    umpt!(solver.g, solver.g[πr.πg])
  end
  solver.μ = computeMu(m, solver; π = πr)
  return g_recompute
end

""" 
    selectPif!(m::FPMPNLPModel, solver::MPR2Solver{T,H}, ωfBound::H)

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

# Modified arguments:
  * `solver.π.πf⁺`: updated with FP format index chosen for objective evaluation
"""
function selectPif!(m::FPMPNLPModel, solver::MPR2Solver{T, H}, ωfBound::H) where {T, H}
  πmin_no_ov = findfirst(x -> x > abs(solver.f - solver.ΔT), m.OFList) # lowest precision level such that predicted f(ck) ≈ fk+gk'ck does not overflow
  if πmin_no_ov === nothing
    solver.π.πf⁺ = solver.πmax
    return
  end
  πmin = max(solver.π.πc, πmin_no_ov) # lower bound on πf⁺ to avoid casting error on c and possible overflow
  f⁺_pred = solver.f - solver.ΔT
  ωf⁺_pred = 0
  if m.ObjEvalMode == REL_ERR
    ωf⁺_pred = abs(f⁺_pred) .* m.ωfRelErr
  elseif m.ObjEvalMode == INT_ERR
    r = abs(f⁺_pred) / abs(solver.f)
    ωf⁺_pred = solver.ωf * r * m.UList ./ m.UList[solver.π.πf]
  else
    solver.π.πf⁺ = πmin
    return
  end
  πf⁺_pred = findfirst(x -> x < ωfBound, ωf⁺_pred)
  solver.π.πf⁺ = πf⁺_pred === nothing ? solver.πmax : max(πf⁺_pred, πmin)
end

"""
    selectPic_default!(π::MPR2Precisions)

Default strategy for selecting FP format of candidate for the next evaluation. Updates `solver.π.πf⁺`.
"""
function selectPic_default!(solver::MPR2Solver)
  solver.π.πc = max(1, solver.π.πf⁺ - 1)
end

####### Default callback function for objective and gradient evaluation #########
"""
    compute_f_at_c_default!(m::FPMPNLPModel, solver::MPR2Solver{T,H}, stats::GenericExecutionStats, e::E)

Compute objective function at the candidate. Updates related fields of solver.

# Outputs:
* `::bool `: returns false if couldn't reach sufficiently small evaluation error or overflow occured. 

"""
function compute_f_at_c_default!(
  m::FPMPNLPModel,
  solver::MPR2Solver{T, H},
  stats::GenericExecutionStats,
  e::E,
) where {H, E, T <: Tuple}
  ωfBound = solver.p.η₀ * solver.ΔT
  selectPif!(m, solver, ωfBound) # select precision evaluation 
  solver.f⁺, solver.ωf⁺, solver.π.πf⁺ = objReachPrec(m, solver.c, ωfBound, π = solver.π.πf⁺)
  if isinf(solver.f⁺) || isinf(solver.ωf⁺)
    @warn "Objective evaluation or error at c overflow"
    return false
  end
  if solver.ωf⁺ > ωfBound
    @warn "Objective evaluation error at c too big"
    return false
  end
  return true
end

"""
    compute_f_at_x_default!(m::FPMPNLPModel, solver::MPR2Solver{T,H}, stats::GenericExecutionStats, e::E)

Compute objective function at the current incumbent. Updates related fields of solver.

# Outputs:
* `::bool` : returns false if couldn't reach sufficiently small evaluation error or overflow occured. 

"""
function compute_f_at_x_default!(
  m::FPMPNLPModel,
  solver::MPR2Solver{T, H},
  stats::GenericExecutionStats,
  e::E,
) where {H, E, T <: Tuple}
  if solver.init == true # first objective eval before main loop, no error bound needed
    solver.f, solver.ωf, solver.π.πf = objReachPrec(m, solver.x, m.OFList[end], π = solver.π.πf)
  else
    ωfBound = solver.p.η₀ * solver.ΔT
    if solver.ωf > ωfBound
      if solver.π.πf == solver.πmax
        @warn "Objective evaluation error at x too big to ensure convergence"
        return false
      end
      solver.π.πf += 1 # default strategy
      solver.f, solver.ωf, solver.π.πf = objReachPrec(m, solver.x, ωfBound, π = solver.π.πf)
      if isinf(solver.f) || isinf(solver.ωf)
        @warn "Objective evaluation or error overflow at x"
        return false
      end
      if solver.ωf > ωfBound
        @warn "Objective evaluation error at x too big to ensure convergence"
        return false
      end
    end
  end
  return true
end

"""
    compute_g_default!(m::FPMPNLPModel, solver::MPR2Solver{T,H}, stats::GenericExecutionStats, e::E)

Compute gradient at x if solver.init == true (first gradient eval outside of main loop), at c otherwise.

# Outputs:
* `::bool` : always true (needed to comply with callback template)
"""
function compute_g_default!(
  m::FPMPNLPModel,
  solver::MPR2Solver{T, H},
  stats::GenericExecutionStats,
  e::E,
) where {H, E, T <: Tuple}
  if solver.init == true # first grad eval at x before main loop
    solver.π.πg = solver.π.πx
    solver.ωg, solver.π.πg = gradReachPrec!(m, solver.x, solver.g, m.OFList[end], π = solver.π.πg)
  else
    solver.π.πg = solver.π.πc # default strategy
    solver.ωg, solver.π.πg = gradReachPrec!(m, solver.c, solver.g, m.OFList[end], π = solver.π.πg)
  end
  return true
end

"""
    recompute_g_default!(m::FPMPNLPModel, solver::MPR2Solver{T,H}, stats::GenericExecutionStats, e::E)

Increase operation precision levels until sufficiently small μ indicator is achieved.
See also [`computeMu`](@ref), [`recomputeMuPrecSelection!`](@ref), [`recomputeMu!`](@ref)

# Outputs:
* `::bool` : returns false if couldn't reach sufficiently small evaluation error or overflow occured. 
"""
function recompute_g_default!(
  m::FPMPNLPModel,
  solver::MPR2Solver{T, H},
  stats::GenericExecutionStats,
  e::E,
) where {H, E, T <: Tuple}
  g_recompute = false
  n = m.meta.nvar
  βfunc(n::Int, u::H) = max(abs(1 - sqrt(1 - m.γfunc(n + 2, u))), abs(1 - sqrt(1 + m.γfunc(n, u)))) # error bound on euclidean norm
  ##### default choice, can be put in a callback for alternative strategy ####
  solver.π.πnx = min(solver.π.πx + 1, solver.πmax) # use better prec to lower βn and therefore phik therefore muk
  solver.π.πns = min(solver.π.πs + 1, solver.πmax)
  ##################################################
  solver.x_norm = H(norm(solver.x[solver.π.πnx]))
  solver.s_norm = H(norm(solver.s[solver.π.πns]))
  solver.ϕhat = solver.x_norm / solver.s_norm
  solver.ϕ =
    solver.ϕhat * (1 + βfunc(n, m.UList[solver.π.πnx])) / (1 - βfunc(n, m.UList[solver.π.πns]))
  prec = findall(u -> u < 1 / solver.ϕ, m.UList)
  if isempty(prec) # step size too small compared to incumbent
    @warn "Algo stops because the step size is too small compare to the incumbent, addition unstable (due to rounding error or absorbtion) with highest precision level"
    stats.status = :small_step
    return false, false
  end
  solver.μ = computeMu(m, solver)
  while solver.μ > solver.p.κₘ
    πr = copy(solver.π)
    max_prec = recomputeMuPrecSelection!(solver.π, πr, solver.πmax)
    if solver.μ > solver.p.κₘ && max_prec
      @warn "Gradient error too big with maximum precision FP format: μ ($(solver.μ)) > κₘ($(solver.p.κₘ))"
      return g_recompute, false
    end
    g_recompute = recomputeMu!(m, solver, πr)
    update_struct!(solver.π, πr)
  end
  return g_recompute, true
end

end # module
