# Author: D. Monnet

# Multi precision regularization algorithm.

# This work has been supported by the NSERC Alliance grant 544900- 19 in collaboration with Huawei-Canada. 


module MultiPrecisionR2

using ADNLPModels, IntervalArithmetic, NLPModels, Printf, LinearAlgebra, SolverCore

export FPMPNLPModel, MPR2Solver, solve!

include("MPNLPModels.jl")

"""
    mpr2s(MPnlp; kwargs...)
An implementation of the quadratic regularization algorithm with dynamic selection of floating point format for objective and gradient evaluation, robust against finite precision rounding errors.

# Arguments
- `MPnlp::FPMPNLPModel{H}` : Multi precision model, see `FPMPNLPModel`
Keyword agruments:
- `x₀::V = MPnlp.MList[end].meta.x0` : initial guess 
- `par::MPR2Params = MPR2Params(MPnlp.FPList[1],H)` : MPR2 parameters, see `MPR2Params` for details
- `atol::H = H(sqrt(eps(T)))` : absolute tolerance on first order criterion 
- `rtol::H = H(sqrt(eps(T)))` : relative tolerance on first order criterion
- `max_iter::Int = 1000` : maximum number of iteration allowed
- `σmin::T = sqrt(T(MPnlp.EpsList[end]))` : minimal value for regularization parameter. Value must be representable in any of the floating point formats of MPnlp. 
- `verbose::Int=0` : display iteration information if > 0

# Outputs
Returns a `GenericExecutionStats`, see `SolverCore.jl`
`GenericExecutionStats.status` is set to `:exception` 
# Examples
```julia
T = [Float16, Float32]
f(x) = x[1]^2+x[2]^2
x = ones(2)
nlp_list = [ADNLPModel(f,t.(x)) for t in T ]
MPnlp = FPMPNLPModel(nlp_list)
mpr2s(MPnlp) 
```
"""
mutable struct MPR2Solver{T <: Tuple
} # <: AbstractOptimizationSolver
  x::T
  g::T
  s::T
  c::T
end

function MPR2Solver(MPnlp::M) where { M <: FPMPNLPModel }
  nvar = MPnlp.MList[1].meta.nvar
  # ElType = typeof(MPnlp.MList[end].meta.x0[1])
  x = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  s = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  c = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  g = Tuple(Vector{ElType}(undef,nvar) for ElType in MPnlp.FPList)
  return MPR2Solver(x, s, c, g)
end


@doc (@doc MPR2Solver) function mpr2s(
  MPnlp::FPMPNLPModel;
  x₀::V = MPnlp.MList[end].meta.x0,
  kwargs...
) where V
  solver = MPR2Solver(MPnlp)
  return solve!(solver, MPnlp;x₀ = x₀, kwargs...)
end

"""
Update multi precision containers. 
Update is occuring only if precision input vector y is lower or equal to the one of the element of the container (vector) to avoid rounding error due to conversion. 
"""
function umpt!(x::Tuple, y::Vector{S}) where {S}
  for xel in x 
    eps(typeof(xel[1])) <= eps(S) && (xel .= y)
  end
end

"""
MPR2 parameters.

# Fields
- `η₀::H` : controls objective function error tolerance, convergence condition is ωf ≤ η₀ ΔT (see `FPMPNLPModel` for details on ωf)
- `η₁::H` : step successful if ρ ≥ η₁ (update incumbent)
- `η₂::H` : step very successful if ρ ≥ η₂ (decrease σ ⟹ increase step length)
- `κₘ::H` : tolerance on gradient evaluation error, μ ≤ κₘ (see `computeMu`) 
- `γ₁::L` : σk+1 = σk * γ₁ if ρ ≥ η₂
- `γ₂::L` : σk+1 = σk * γ₂ if ρ < η₁

# Parameters
- `H` must correspond to `MPnlp.HPFormat` with `MPnlp` given as input of `mpr2s`
- `L` must correspond to `MPnlp.FPList[1]`, i.e the lowest precision floating point format used by `MPmodel` given as input of `mpr2s`

# Conditions 
Parameters must statisfy the following conditions:
* 0 ≤ η₀ ≤ 1/2*η₁
* 0 ≤ η₁ ≤ η₂ < 1
* η₀+κₘ/2 ≤0.5*(1-η₂)
* η₂<1 
* 0<γ₁<1<γ₂

# Constructor
`MPR2Params(LPFormat::DataType,HPFormat::DataType)` : 
- `LPFormat` : lowest precision FP format of the FPMPNLPModel to be solved
- `HPFormat` : high precision format of the FPMPNLPModel to be solved

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
Precision of variables and precision evaluation of obj and grad.
Precisions are represented as integer, and correspond to FP format of corresponding index in FPList of FPMPNLPModel.
See [`FPMPNLPModel`](@ref)
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


function solve!(
  solver::MPR2Solver{T},
  MPnlp::FPMPNLPModel{H};
  x₀::V = MPnlp.MList[end].meta.x0,
  par::MPR2Params = MPR2Params(MPnlp.FPList[1],H),
  atol::H = H(sqrt(eps(MPnlp.FPList[end]))),
  rtol::H = H(sqrt(eps(MPnlp.FPList[end]))),
  max_iter::Int = 1000,
  σmin::S = sqrt(MPnlp.FPList[end](MPnlp.EpsList[end])),
  verbose::Int=0
) where {S, V <:Vector{S}, H, T}
  start_time = time()
  elapsed_time = 0.0
  CheckMPR2ParamConditions(par)
  #initialize parameters
  η₀ = par.η₀
  η₁ = par.η₁
  η₂ = par.η₂
  κₘ = par.κₘ
  γ₁ = par.γ₁ # in lowest precision format to ensure type stability when updating σ
  γ₂ = par.γ₂ # in lowest precision format to ensure type stability when updating σ
  #initialize variables
  umpt!(solver.x, x₀)
  x = solver.x
  g = solver.g
  s = solver.s
  c = solver.c
  #initialize precisions
  πmax = length(MPnlp.MList)
  π = MPR2Precisions(πmax)
  #initialize formats precisions
  n = MPnlp.MList[1].meta.nvar
  U = H(1/2)*MPnlp.EpsList #assuming unit round-off is 1/2 machine ϵ (RN rounding mode)
  FP = MPnlp.FPList
  OF = H.(prevfloat.(typemax.(FP))) #overflow value list
  #dot product and norm error bound functions
  γfunc = MPnlp.γfunc 
  αfunc(n::Int,u::H) = 1/(1-γfunc(n,u))
  βfunc(n::Int,u::H) = max(abs(1-sqrt(1-γfunc(n+2,u))),abs(1-sqrt(1+γfunc(n,u)))) # error bound on euclidean norm
  #misc initialization
  iter=0
  done = false
  status = :exception
  # initial evaluation, check for overflow
  f, ωf, π.πf = objReachPrec(MPnlp, x, OF[end], π = π.πf)
  if f === FP[end](Inf) ||  ωf === FP[end](Inf)
    @warn "Objective or ojective error overflow at x0"
    status = :exception
    done = true
  end
  _, ωg, π.πg = gradReachPrec!(MPnlp, x, g, OF[end], π = π.πg)
  # g.=gout # keep g type consistent
  if findfirst(x->x === FP[end](Inf), g) !== nothing || ωg == FP[end](Inf)
    @warn "Objective or ojective error overflow at x0"
    status = :exception
    done = true
  end
  g_norm = norm(g[π.πg])
  σ0 = 2^round(log2(g_norm+1)) # ensures ||s_0|| ≈ 1 with sigma_0 = 2^n with n an interger, i.e. sigma_0 is exactly representable in n bits 
  σ = H(σ0) # declare σ with the highest precision, convert it when computing sk to keep it at precision πg
  # check first order stopping condition
  ϵ = atol + rtol * g_norm
  if g_norm ≤ (1-βfunc(n,U[π.πg]))*ϵ/(1+H(ωg))
    done = true
    status = :first_order
  end
  if verbose ≥ 0
    infoline = @sprintf "%6s  %9s  %9s  %9s  %9s  %9s  %7s  %7s  %7s  %7s %7s  %2s  %2s  %2s  %2s\n" "iter" "f(x)" "ωf(x)" "f(c)" "ωf(c)" "‖g‖" "ωg" "σ" "μ" "ϕ" "ρ" "πx" "πc" "πf" "πg"
    @info infoline
  end
  muFail = false
  #main loop
  while(!done)
    iter += 1
    computeStep!(s, g, σ, FP, π)
    if isinf(s[1][1]) # overflow or underflow occured, stop the loop
      @warn "Step over/underflow"
      status = :exception
      break
    end
    computeCandidate!(c, x, s, FP, π)
    if isinf(c[1][1]) # overflow occured, stop the loop
      @warn "Candidate over/underflow"
      status = :exception
      break
    end
    ΔT = H(computeModelDecrease!(g, s, FP, π))
    if ΔT == Inf || ΔT == 0.0 
      @warn "Model decrease over/underflow"
      status = :exception
      break
    end
    ##### default choice, can be put in a callback for alternative strategy ####
    π.πnx = min(π.πx+1,πmax) # use better prec to lower βn and therefore phik therefore muk
    π.πns = min(π.πs+1,πmax) 
    ##################################################
    x_norm = H(norm(x[π.πnx]))
    # x_norm = H(norm(FP[π.πnx].(x)))
    s_norm = H(norm(s[π.πns]))
    # s_norm = H(norm(FP[π.πns].(s)))
    ϕhat = x_norm/s_norm
    ϕ = ϕhat * (1+βfunc(n,U[π.πnx]))/(1-βfunc(n,U[π.πns]))
    prec = findall(u->u<1/ϕ,U) # step size too small compared to incumbent
    if isempty(prec)
      @warn "Algo stops because the step size is too small compare to the incumbent, addition unstable (due to rounding error or absorbtion) with highest precision level"
      status = :small_step
      break
    end
    μ = computeMu(MPnlp, U, n, ωg, ϕ, π, αfunc)
    πr = copy(π)
    while μ > κₘ 
      recomputeMuPrecSelection!(π, πr, πmax) # default strategy, could be a callback
      if μ > κₘ && πr.πx == -1
        @warn "Gradient error too big with maximum precision FP format: μ ($μ) > κₘ($κₘ)"
        muFail = true
        break
      end
      μ, ωg = recomputeMu!(MPnlp, x, g, s, x_norm, s_norm, ΔT, n, ϕ, ωg, π, πr ,U, αfunc, βfunc)
      π = copy(πr)
    end
    muFail == false || break
    ωfBound = η₀*ΔT
    if ωf > ωfBound
      if π.πf == πmax
        @warn "Objective evaluation error at x too big to ensure convergence"
        status = :exception
        break
      end
      π.πf += 1 # default strategy, could be a callback
      f, ωf, π.πf = objReachPrec(MPnlp, x, ωfBound, π = π.πf)
      if f === FP[end](Inf) ||  ωf === FP[end](Inf)
        @warn "Objective evaluation error at x too big to ensure convergence"
        status = :exception
        break
      end
    end
    selectPif!(MPnlp, f, ωf, ΔT, ωfBound, OF, U, π) # default strategy, could be a callback
    f⁺, ωf⁺, π.πf⁺ = objReachPrec(MPnlp, c, ωfBound, π = π.πf⁺)
    if f⁺ === FP[end](Inf) ||  ωf⁺ === FP[end](Inf)
      @warn "Objective evaluation error at c too big to ensure convergence"
      status = :exception
      break
    end
    ρ = ( H(f) - H(f⁺) ) / H(ΔT)
    if verbose ≥ 0
      infoline = @sprintf "%6d  %9.2e  %9.2e  %9.2e  %9.2e  %9.2e  %7.1e  %7.1e  %7.1e  %7.1e  %7.1e  %2d  %2d  %2d  %2d \n" iter f ωf f⁺ ωf⁺ g_norm ωg σ μ ϕ ρ π.πx π.πc π.πf π.πg
      @info infoline
    end
    if ρ ≥ η₂ 
      σ = max(σmin,  σ*γ₁)
    end
    if ρ < η₁
      σ = σ * γ₂
    end
    if ρ ≥ η₁
      π.πg = π.πc # default strategy, could be a callback
      _, ωg, π.πg = gradReachPrec!(MPnlp, c, g, OF[end], π = π.πg)
      umpt!(g,g[π.πg])
      # gout, ωg, π.πg = gradReachPrec!(MPnlp, c, g, OF[end], π = π.πg)
      # g.=gout
      if findfirst(x->x === FP[end](Inf), g[end]) !== nothing || ωg == FP[end](Inf)
      #if findfirst(x->x === FP[end](Inf), g) !== nothing || ωg == FP[end](Inf)
        @warn "gradient evaluation error at c too big to ensure convergence"
        status = :exception
        break
      end
      g_norm = H(norm(g[π.πg]))
      # g_norm = H(norm(g))
      umpt!(x,c[π.πc])
      # x .= c
      f = f⁺
      ωf = ωf⁺
      π.πf = π.πf⁺
      π.πx = π.πc
      selectPic!(π)
    end
    if g_norm ≤1/(1+βfunc(n,U[π.πg]))*ϵ/(1+H(ωg))
      status = :first_order
      done = true
    end
    if iter > max_iter
      status = :max_iter
      done = true
    end
  end
  elapsed_time = time() - start_time
  return GenericExecutionStats(
    MPnlp.MList[end],
    status = status,
    solution = x[end],
    # solution = x,
    objective = MPnlp.FPList[end](f),
    dual_feas = MPnlp.FPList[end](g_norm),
    elapsed_time = elapsed_time,
    iter = iter,
  )
end

"""
Compute μ value for gradient error ωg, ratio ϕ = ||x||/||s|| and rounding error models
"""
function computeMu(m::FPMPNLPModel{H}, U::Vector{H}, n::Int, ωg, ϕ::H, π::MPR2Precisions, αfunc ) where H
  u = π.πc >= π.πs ? U[π.πc] : U[π.πc] + U[π.πs] + U[π.πc]*U[π.πs]
  return ( αfunc(n+1,U[π.πΔ]) * H(ωg) * (U[π.πc] + ϕ * u +1) + αfunc(n+1,U[π.πΔ]) * u * (ϕ +1) + U[π.πg] + m.γfunc(n+2,U[π.πΔ]) * αfunc(n+1,U[π.πΔ]) ) / (1-U[π.πg])
end

""" Default strategy to select new precisions to recompute μ in the case where μ > κₘ
Takes as input all values and functions needed to compute muFail.
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
  elseif π.πc < πmax
    πr.πc = π.πc +1
  # Priority #4 strategy: recompute gradient
  elseif π.πg < πmax
    πr.πg = π.πg+1
  else
    πr.πx = -1 # max precision reached
  end
  return πr
end

""" Recompute mu based on new precision levels. 
Performs only necessary steps of solve! main loop to recompute mu. 
Possible step to recompute are:
- recompute ϕhat and ϕ with higher FP format for norm computation of x and s
- recompute gradient with higher precision to decrease ωg
- recompute candidate with higher prec FP format to decrease u
- recompute model reduction with higher precision to decrease αfunc(n,U[π.πΔ])
Does not make the over/underflow check as in main loop, since it is a repetition of the main loop with higher precisions and these issue shouldn't occur
# Outputs:
* μ::H : new value of μ
* ωg::H : new gradient error if gradient is recomputed

See [`recomputeMuPrecSelection`](@ref)
"""
function recomputeMu!(m::FPMPNLPModel{H}, x::T, g::T, s::T, x_norm::H, s_norm::H, ΔT::H, n::Int, ϕ::H, ωg::H, π::MPR2Precisions, πr::MPR2Precisions, U::Vector{H}, αfunc, βfunc) where {T <: Tuple, H}
  ΔTnew = ΔT
  ϕnew = ϕ
  ωgnew = ωg
  x_norm_new = x_norm
  s_norm_new = s_norm
  if π.πΔ != πr.πΔ # recompute model decrease
    ΔTnew = computeModelDecrease!(g, s, FP, πr)
  end
  if π.πnx != πr.πnx || π.πns != πr.πns # recompute x, s norm and ϕ
    π.πnx != πr.πnx ? x_norm_new = H(norm(x[πr.πnx])) : x_norm
    π.πns != πr.πns ? s_norm_new = H(norm(s[πr.πns])) : s_norm
    ϕhat_new = x_norm/s_norm
    ϕnew = ϕhat_new * (1+βfunc(n,U[πr.πnx]))/(1-βfunc(n,U[πr.πns]))
  end
  if π.πg != πr.πg
    gout, ωgnew = graderrmp!(m, x[πr.πg], g[πr.πg], πr.πg) 
    umpt!(g,gout)
  end
  μnew = computeMu(m, U, n, ωgnew, ϕnew, πr, αfunc)
  return μnew, H(ωgnew)
end

"""
Compute step with FP format avoiding underflow and overflow
# Arguments
* `g::Vector{T}` : gradient 
* `πg::Int` : `g` FP index
* `σ::H` : regularization parameter
* `FP::Vector{Int}` : Available floating point formats

Modified :
* `s::Vector` : step vector container
* `π::MPR2Precisions` : hold the FP format indices
# Output
* `s::Vector{B}` : step computed as `-g`/`σ` that doesn't underflow
* `π::Int`  
If underflow or overflow happens with the highest precision FP format (FP[end]), Inf vector is returned
"""
function computeStep!(s::T, g::T, σ::H, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple, H}
  πmax = length(FP)
  πs = π.πg
  while FP[πs](σ) === FP[πs](0) || FP[πs](σ) === FP[πs](Inf) # σ cast underflow or overflow
    πs += 1
    if πs == πmax +1 # over/underflow at max prec FP
      @error "over/underflow of σ with maximum precision FP format ($(FP[πmax]))"
      π.πs = πmax
      umpt!(s,FP[1].([Inf for _ in eachindex(s[1])]))
      return s, π
    end
  end
  sval = - g[πs]/FP[πs](σ)
  umpt!(s, sval)
  # s .= - FP[πs].(g)/FP[πs](σ)# ::Vector{FP[πs]}, Vector{T} in memory
  # increase FP prec until no overflow or underflow occurs
  while findall(x->x==0.0,sval) != findall(x->x==0.0,g[π.πg]) || findfirst(x->isinf(x),sval) !== nothing 
  # while findall(x->x==0.0,s) != findall(x->x==0.0,g) || findfirst(x->isinf(x),s) !== nothing 
    if πs == πmax #not enough precision to avoid over/underflow 
      @error "over/underflow with maximum precision FP format ($(FP[πmax]))"
      umpt!(s, FP[1].([ Inf for _ in eachindex(s[1])] )) 
      # s .= [Inf for _=1:length(s)]
      π.πs = πs
      break
    end
    πs +=1 
    sval = - g[πs] / FP[πs](σ)
    umpt!(s, sval)
    # s .= - FP[πs].(g)/FP[πs](σ) # ::Vector{FP[πs]}
  end
  π.πs = πs
  return s, π
end

"""
Compute candidate with FP format avoiding underflow and overflow
###### Inputs
* x::Vector{T} : incumbent 
* s::Vector{T} : step
* FP::Vector{Int} : Available floating point formats

Modified:
* c::Vector{T} : candidate
* π::MPR2Precisions : hold the FP format indices
###### Output
* c::Vector{B} : candidate computed as x+s that doesn't overflow
* πc:: Int : FP[πc] = B 
If underflow or overflow happens with the highest precision FP format (FP[end]), Inf vector is returned
"""
function computeCandidate!(c::T, x::T, s::T, FP::Vector{DataType}, π::MPR2Precisions) where {T <: Tuple}
  πmax = length(FP)
  πx = π.πx
  πs = π.πs
  πc = π.πc
  cval = FP[πc].(x[max(πc,πx)] .+ s[max(πc,πs)])
  umpt!(c, cval)
  # c .= FP[πc].(FP[max(πc,πx)].(x) .+ FP[max(πc,πs)].(s))
  while findfirst(x->x==Inf,cval) !== nothing || c[πc] == x[πx]
    if πc == πmax #not enough precision to avoid underflow
      @warn "over/underflow with maximum precision FP format ($(FP[πmax]))"
      umpt!(c, FP[1].([Inf for _ in eachindex(c[1])])) 
      # c .= [Inf for _=1:length(c)]
      π.πc = πc
      break
    end
    πc +=1 
    cval = FP[πc].(x[max(πc,πx)] .+ s[max(πc,πs)])
    umpt!(c, cval)
    # c .= FP[πc].(FP[max(πc,πx)].(x) .+ FP[max(πc,πs)].(s))
  end
  π.πc = πc
  return c, π
end

"""
Compute model decrease with FP format avoiding underflow and overflow
###### Inputs
* s::Vector{T} : step
* g::Vector{T} : gradient 
* FP::Vector{Int} : Available floating point formats

Modified
* π::MPR2Precisions : hold the FP format indices
###### Output
* ΔT::FP[π.πc] : step computed as -g/σ that doesn't underflow
* π:: Int : FP[πc] = B 
If overflow happens with the highest precision FP format (FP[end]), Inf vector is returned
"""
function computeModelDecrease!(g::T,s::T,FP::Vector{DataType},π::MPR2Precisions) where {T <: Tuple}
  πΔ = π.πΔ
  πmax = length(FP)
  if πΔ < max(π.πs,π.πg)
    error("Model decrease computation FP format should be greater that FP format of g and s")
  end
  ΔT = - dot(g[πΔ], s[πΔ]) 
  # ΔT = - dot(FP[πΔ].(g), FP[πΔ].(s)) # allocation because of casting 
  while isinf(ΔT) || ΔT == 0.0
    if πΔ == πmax
      @warn "over/underflow with maximum precision FP format ($(FP[πmax]))"
      break
    end
    πΔ += 1
    ΔT = - dot(g[πΔ], s[πΔ]) 
    # ΔT = - dot(FP[πΔ].(g), FP[πΔ].(s))
  end
  π.πΔ = πΔ
  return ΔT
end

"""
Evaluates objective and increase model precision to reach necessary error bound.
##### Inputs
* `π`: Initial ''guess'' for NLPModel in m.MList that can provide evaluation error lower than `err_bound`, use 1 by default (lowest precision)
##### Outputs
* `f`: objective value at `x`
* `ωf`: objective evaluation error
* `id`: id-th model(`m.MList[id]`) provided `ωf ≤ err_bound`.
There is no guarantee that `ωf ≤ err_bound`, happens if highest precision model (`m.MList[end]`) is not accurate enough.
If overflow occurs with highest precision model(`m.MList[end]`), see [`objerrmp`](@ref) for returned values
"""
function objReachPrec(m::FPMPNLPModel{H}, x::T, err_bound::H; π::Int = 1) where {T <: Tuple, H}
  id = π
  πmax = length(m.MList)
  f, ωf = objerrmp(m,x[id],id)
  while ωf > err_bound && id ≤ πmax-1
    id += 1
    f, ωf = objerrmp(m,x[id],id)
  end
  if id == πmax && f === m.FPList[πmax](Inf)
    @warn "Objective evaluation overflows with highest FP format at x0"
  end
  if id == πmax && ωf === m.FPList[πmax](Inf)
    "Objective evaluation overflows with highest FP format at x0"
  end
  return H(f), H(ωf), id
end

"""
Evaluates gradient and increase model precision until necessary error bound is reached.
# Inputs
* `π`: Initial ''gess'' for NLPModel in m.MList that can provide evaluation error lower than `err_bound`, use 1 by default (lowest precision)
# Outputs
* `g`: gradient vector at `x`
* `ωg`: objective evaluation error
* `id`: id-th model(`m.MList[id]`) provided `ωg ≤ err_bound`.
There is no guarantee that `ωg ≤ err_bound`, happens if highest precision model (`m.MList[end]`) is not accurate enough.
If overflow occurs with highest precision model(`m.MList[end]`), see [`objerrmp`](@ref) for returned values
"""
function gradReachPrec!(m::FPMPNLPModel{H}, x::T, g::T, err_bound::H; π::Int = 1) where {T <: Tuple, H}
  id = π
  πmax = length(m.MList)
  _, ωg = graderrmp!(m, x[id], g[id], id)
  umpt!(g,g[id])
  # fp = m.FPList[id]
  # g, ωg = graderrmp(m,fp.(x),id)
  while ωg > err_bound && id ≤ πmax-1
    id += 1
    _, ωg = graderrmp!(m, x[id], g[id], id)
    umpt!(g, g[id])
    # fp = m.FPList[id]
    # g, ωg = graderrmp(m,fp.(x),id)
  end
  if findfirst(x->x==Inf,g) !== nothing
    @warn "Gradient evaluation overflows with highest FP format at x0"
  end
  if id == πmax && ωg === m.FPList[id](Inf)
    "Gradient evaluation overflows with highest FP format at x0"
  end
  return g, H(ωg), id
end

@doc (@doc gradReachPrec!)
function gradReachPrec(m::FPMPNLPModel{H}, x::T, err_bound::H; π::Int = 1) where {T <: Tuple, H}
  nvar = length(x[1])
  g = Tuple(Vector{ElType}(undef,nvar) for ElType in m.FPList)
  return gradReachPrec!(m, x, g, err_bound, π = π)
end


  
""" Select a precision for objective evaluation for candidate based on predicted evaluation error.
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
function selectPif!(m::FPMPNLPModel{H}, f::H, ωf::H, ΔT::H, ωfBound::H, OF::Vector{H}, U::Vector{H}, π::MPR2Precisions) where H
  πmax = length(m.MList)
  πmin_no_ov = findfirst(x -> x > abs(f) - ΔT, OF) # lowest precision level such that predicted f(ck) ≈ fk+gk'ck does not overflow
  if isempty(πmin_no_ov)
    π.πf⁺ = πmax
    return π
  end
  πmin = max(π.πc,πmin_no_ov) # lower bound on πf⁺ to avoid casting error on c and possible overflow
  f⁺_pred = f-ΔT 
  ωf⁺_pred = 0
  if m.ObjEvalMode == REL_ERR
    ωf⁺_pred = f⁺_pred .* m.ωfRelErr
  elseif m.ObjEvalMode == INT_ERR 
    r = abs(f⁺_pred)/abs(f)
    ωf⁺_pred = ωf * r * U ./ U[π.πf]
  else 
    π.πf⁺ = πmin
    return π
  end
  πf⁺_pred = findfirst(x -> x<ωfBound, ωf⁺_pred)
  π.πf⁺ = isempty(πf⁺_pred) ? πmax : max(πf⁺_pred,πmin)
  return π
end

function selectPic!(π::MPR2Precisions)
  π.πc = max(1,π.πf⁺-1)
end

"""
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

end # module