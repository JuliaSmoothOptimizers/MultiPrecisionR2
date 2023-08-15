# "A Multi-Precision Quadratic Regularization Method for Unconstrained Optimization with Rounding Error Analysis" Numerical Results


## MRP2 Guaranteed Version
```@example
using MultiPrecisionR2
using NLPModels
using ADNLPModels
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using Quadmath
using IntervalArithmetic

setrounding(Interval,:accurate)
FP = [Float16,Float32,Float64] # MPR2 Floating Point formats
mpr2_obj_eval = zeros(length(FP))
mpr2_grad_eval = zeros(length(FP))
mpr2_status = Dict{Symbol,Int}()
nvar = 10 #problem dimension (if scalable)
max_iter = 10
param = MPR2Params(Float128.([0.1,0.3,0.7,0.1])...,Float16(1/2),Float16(2))

mpmodel = nothing

meta = OptimizationProblems.meta
names_pb_vars = meta[(meta.has_bounds .== false) .& (meta.ncon .== 0), [:nvar, :name]] #select unconstrained problems
for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(n=$nvar,type=Val(Float64),backend = :generic)"))
  @show nlp.meta.name
  try
    mpmodel = FPMPNLPModel(nlp,FP,HPFormat = Float128,obj_int_eval=true,grad_int_eval=true); # instanciation checks might error in some case due to over/underflow with low-prec/low-range formats
  catch e
    continue
  end
  statmpr2 = MPR2(mpmodel,max_iter = max_iter,run_free = true,par = param)
  mpr2_obj_eval .+= [haskey(mpmodel.counters.neval_obj,fp) ? mpmodel.counters.neval_obj[fp] : 0 for fp in FP]
  mpr2_grad_eval .+= [haskey(mpmodel.counters.neval_grad,fp) ? mpmodel.counters.neval_grad[fp] : 0 for fp in FP]
  haskey(mpr2_status,statmpr2.status) ? mpr2_status[statmpr2.status] +=1 : mpr2_status = merge(mpr2_status,Dict(statmpr2.status => 1))
end
```

## relaxed-MPR2 vs R2


```@example
using MultiPrecisionR2
using NLPModels
using ADNLPModels
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using JSOSolvers
using Quadmath

FP = [Float16,Float32,Float64] # MPR2 Floating Point formats
omega = Float128.(sqrt.(eps.(FP)))
r2_obj_eval = [0]
r2_grad_eval = [0]

mu_factor = Float128.([1,0.1,0.01])
mpr2_obj_eval = [zeros(length(FP)) for _ in eachindex(mu_factor)]
mpr2_grad_eval = [zeros(length(FP)) for _ in eachindex(mu_factor)]
r2_status = Dict{Symbol,Int}()
mpr2_status = [Dict{Symbol,Int}() for _ in eachindex(mu_factor)]
nvar = 200 #problem dimension (if scalable)
max_iter = 10000
gamma(n,u) = sqrt(n)*u
param = MPR2Params(Float128.([0.05,0.1,0.7,0.2])...,Float16(1/2),Float16(2))

mpr2_error_count=0
mpmodel = nothing

meta = OptimizationProblems.meta
names_pb_vars = meta[(meta.has_bounds .== false) .& (meta.ncon .== 0), [:nvar, :name]] #select unconstrained problems

for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(n=$nvar,type=Val(Float64),backend = :generic)"))
  @show nlp.meta.name
  try
    mpmodel = FPMPNLPModel(nlp,FP,HPFormat = Float128,γfunc = gamma,ωfRelErr = omega,ωgRelErr = omega); # instanciation checks might error in some case due to over/underflow with low-prec/low-range formats
  catch e
    continue
  end
  for i in eachindex(mu_factor)
    statmpr2 = MPR2(mpmodel,max_iter = max_iter,run_free = true,par = param,mu_factor = mu_factor[i])
    mpr2_obj_eval[i] .+= [haskey(mpmodel.counters.neval_obj,fp) ? mpmodel.counters.neval_obj[fp] : 0 for fp in FP]
    mpr2_grad_eval[i] .+= [haskey(mpmodel.counters.neval_grad,fp) ? mpmodel.counters.neval_grad[fp] : 0 for fp in FP]
    haskey(mpr2_status[i],statmpr2.status) ? mpr2_status[i][statmpr2.status] +=1 : mpr2_status[i] = merge(mpr2_status[i],Dict(statmpr2.status => 1))
  end
  
  reset!(nlp)
  statr2 = R2(nlp,max_eval=max_iter,η1=0.1,η2 =0.7)
  r2_obj_eval .+= nlp.counters.neval_obj
  r2_grad_eval .+= nlp.counters.neval_grad
  haskey(r2_status,statr2.status) ? r2_status[statr2.status] +=1 : r2_status = merge(r2_status,Dict(statr2.status => 1))
end
data_header = ["mu factor","obj time effort","obj energy effort","grad time effort","grad energy effort"]
data_mpr2 = Tuple(Vector{Float64}(undef,3) for _ in 1:5)
data_mpr2[1] .= mu_factor

mpr2_obj_time = sum.([mpr2_obj_eval[j].*reverse([1.0/2^(i-1) for i in eachindex(FP)]) for j in eachindex(mu_factor)])
data_mpr2[2] .= mpr2_obj_time./r2_obj_eval[1]
mpr2_obj_energy = sum.([mpr2_obj_eval[j].*reverse([1.0/4^(i-1) for i in eachindex(FP)]) for j in eachindex(mu_factor)])
data_mpr2[3] .= mpr2_obj_energy./r2_obj_eval[1]
mpr2_grad_time = sum.([mpr2_grad_eval[j].*reverse([1.0/2^(i-1) for i in eachindex(FP)]) for j in eachindex(mu_factor)])
data_mpr2[4] .= mpr2_grad_time./r2_grad_eval[1]
mpr2_grad_energy = sum.([mpr2_grad_eval[j].*reverse([1.0/4^(i-1) for i in eachindex(FP)]) for j in eachindex(mu_factor)])
data_mpr2[5] .= mpr2_grad_energy./r2_grad_eval[1]


println("Possible time saving for objective evaluation with MPR2: $(round((1-obj_time_save[1])*100,digits=1)) %")
println("Possible time saving for gradient evaluation with MPR2: $(round((1-grad_time_save[1])*100,digits=1)) %")
println("Possible energy saving for objective evaluation with MPR2: $(round((1-obj_energy_save[1])*100,digits=1)) %")
println("Possible energy saving for gradient evaluation with MPR2: $(round((1-grad_energy_save[1])*100,digits=1)) %")
```

```@example
using MultiPrecisionR2
using NLPModels
using ADNLPModels
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using JSOSolvers
using Quadmath

FP = [Float16,Float32,Float64] # MPR2 Floating Point formats
omega = Float128.((eps.(FP)))
r2_obj_eval = [0]
r2_grad_eval = [0]
mpr2_obj_eval = zeros(length(FP))
mpr2_grad_eval = zeros(length(FP))
r2_status = Dict{Symbol,Int}()
mpr2_status = Dict{Symbol,Int}()
nvar = 200 #problem dimension (if scalable)
max_iter = 10000
gamma(n,u) = sqrt(n)*u
param = MPR2Params(Float128.([0.05,0.1,0.7,0.2])...,Float16(1/2),Float16(2))


mpr2_error_count=0
mpmodel = nothing

meta = OptimizationProblems.meta
names_pb_vars = meta[(meta.has_bounds .== false) .& (meta.ncon .== 0), [:nvar, :name]] #select unconstrained problems
pb_issue = []
for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(n=$nvar,type=Val(Float64),backend = :generic)"))
  @show nlp.meta.name
  try
    mpmodel = FPMPNLPModel(nlp,FP,HPFormat = Float128,γfunc = gamma,ωfRelErr = omega,ωgRelErr = omega); # instanciation checks might error in some case due to over/underflow with low-prec/low-range formats
  catch e
    continue
  end
  statmpr2 = MPR2(mpmodel,max_iter = max_iter,run_free = true,par = param)
  mpr2_obj_eval .+= [haskey(mpmodel.counters.neval_obj,fp) ? mpmodel.counters.neval_obj[fp] : 0 for fp in FP]
  mpr2_grad_eval .+= [haskey(mpmodel.counters.neval_grad,fp) ? mpmodel.counters.neval_grad[fp] : 0 for fp in FP]
  haskey(mpr2_status,statmpr2.status) ? mpr2_status[statmpr2.status] +=1 : mpr2_status = merge(mpr2_status,Dict(statmpr2.status => 1))

  reset!(nlp)
  statr2 = R2(nlp,max_eval=max_iter,η1=0.1,η2 =0.7)
  r2_obj_eval .+= nlp.counters.neval_obj
  r2_grad_eval .+= nlp.counters.neval_grad
  haskey(r2_status,statr2.status) ? r2_status[statr2.status] +=1 : r2_status = merge(r2_status,Dict(statr2.status => 1))
end
mpr2_obj_time = sum(mpr2_obj_eval.*reverse([1.0/2^(i-1) for i in eachindex(FP)]))
obj_time_save = mpr2_obj_time/r2_obj_eval[1]
mpr2_obj_energy = sum(mpr2_obj_eval.*reverse([1.0/4^(i-1) for i in eachindex(FP)]))
obj_energy_save = mpr2_obj_energy/r2_obj_eval[1]
mpr2_grad_time = sum(mpr2_grad_eval.*reverse([1.0/2^(i-1) for i in eachindex(FP)]))
grad_time_save = mpr2_grad_time/r2_grad_eval[1]
mpr2_grad_energy = sum(mpr2_grad_eval.*reverse([1.0/4^(i-1) for i in eachindex(FP)]))
grad_energy_save = mpr2_grad_energy/r2_grad_eval[1]
println("Possible time saving for objective evaluation with MPR2: $(round((1-obj_time_save)*100,digits=1)) %")
println("Possible time saving for gradient evaluation with MPR2: $(round((1-grad_time_save)*100,digits=1)) %")
println("Possible energy saving for objective evaluation with MPR2: $(round((1-obj_energy_save)*100,digits=1)) %")
println("Possible energy saving for gradient evaluation with MPR2: $(round((1-grad_energy_save)*100,digits=1)) %")
```
