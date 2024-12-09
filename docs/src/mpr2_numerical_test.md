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
using DataFrames
using PrettyTables
using CSV

setrounding(Interval,:accurate)
FP = [Float16,Float32,Float64] # MPR2 Floating Point formats
HP_format = Float128 # High precision format

# MPR2 parameters
η0 = 0.05
η1=0.1
η2 =0.7
κm = 0.2
γ1 = 1/2
γ2 = 2.0
max_iter = 20000
max_time = 900.0
atol = eps(FP[end])^(1/4)
rtol = atol

param = MPR2Params(HP_format.([η0, η1, η2, κm])...,Float16(γ1),Float16(γ2))

mpmodel = nothing

df_str = ":name => String[], :status => Symbol[]"
col_str = [":neval_obj_",":neval_obj_fail_",":neval_grad_",":neval_grad_fail_"]
for col in col_str
  for fp in FP
    df_str *= ","*col*"$fp => Int64[]"
  end
end
stats = eval(Meta.parse("DataFrame($df_str)"))

pb_error_skip = []
meta = OptimizationProblems.meta
names_pb_vars = meta[(meta.has_bounds .== false) .& (meta.ncon .== 0), [:nvar, :name]] #select unconstrained problems
for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=Float64,backend = :generic)"))
  @show nlp.meta.name
  try
    mpmodel = FPMPNLPModel(nlp,FP,HPFormat = HP_format,obj_int_eval=true,grad_int_eval=true); # instanciation checks might error in some case due to over/underflow with low-prec/low-range formats
  catch e
    println("skip $(nlp.meta.name)")
    push!(pb_error_skip,nlp.meta.name)
    continue
  end
  statmpr2 = MPR2(mpmodel,max_iter = max_iter,run_free = false, max_time = max_time, par = param,atol = HP_format(atol), rtol = HP_format(rtol), verbose=1)
  push!(stats,
      [nlp.meta.name,
      statmpr2.status,
      [mpmodel.counters.neval_obj[fp] for fp in FP]...,
      [mpmodel.counters_fail.neval_obj[fp] for fp in FP]...,
      [mpmodel.counters.neval_grad[fp]  for fp in FP]...,
      [mpmodel.counters_fail.neval_grad[fp] for fp in FP]...]
    )
    CSV.write("guaranteed_implem_results_tol_pow_0_25.csv",stats)
end

data_header_table = ["Algo",vcat([["eval_$fp","success_rate"] for fp in FP]...)...,"FO", "MI", "F"]
FO = nrow(filter(row -> row.status == "first_order" || row.status == :first_order,stats))
MI = nrow(filter(row -> row.status == "max_iter" || row.status == :max_iter,stats))
F = nrow(filter(row -> row.status == "exception" || row.status == :exception,stats)) 
MT = nrow(filter(row -> row.status == "max_time" || row.status == :max_time,stats))

obj_eval_fields = [Meta.parse("neval_obj_$(FP[i])") for i in eachindex(FP)]
obj_eval_fail_fields = [Meta.parse("neval_obj_fail_$(FP[i])") for i in eachindex(FP)]

obj_eval = [sum(stats[!,obj_eval_fields[j]]) for j in eachindex(FP)]
obj_eval_fail = [sum(stats[!,obj_eval_fail_fields[j]]) for j in eachindex(FP)]
obj_total_eval = sum([sum(stats[!,obj_eval_fields[j]]) for j in eachindex(FP)])

obj_eval_ratio = [obj_eval[j]/sum(obj_eval) for j in eachindex(FP)]
obj_success_ratio = [(obj_eval[j] - obj_eval_fail[j])/obj_eval[j] for j in eachindex(FP)]

data_obj = ["MPR2",vcat([[obj_eval_ratio[i],obj_success_ratio[i]] for i in eachindex(FP)]...)...,FO,MI,F]
pretty_table(hcat(data_obj...),header = data_header_table)

# grad eval stats
grad_eval_fields = [Meta.parse("neval_grad_$(FP[i])") for i in eachindex(FP)]
grad_eval_fail_fields = [Meta.parse("neval_grad_fail_$(FP[i])") for i in eachindex(FP)]

grad_eval = [sum(stats[!,grad_eval_fields[j]]) for j in eachindex(FP)]
grad_eval_fail = [sum(stats[!,grad_eval_fail_fields[j]]) for j in eachindex(FP)]
grad_total_eval = sum([sum(stats[!,grad_eval_fields[j]]) for j in eachindex(FP)])

grad_eval_ratio = [grad_eval[j]/sum(grad_eval) for j in eachindex(FP)]
grad_success_ratio = [(grad_eval[j] - grad_eval_fail[j])/grad_eval[j] for j in eachindex(FP)]

data_grad = ["MPR2",vcat([[grad_eval_ratio[i],grad_success_ratio[i]] for i in eachindex(FP)]...)...,FO,MI,F]
pretty_table(hcat(data_grad...),header = data_header_table)
```

## relaxed-MPR2 vs R2


```@example mpr2vsr2
using MultiPrecisionR2
using DataFrames
using PrettyTables
using NLPModels
using ADNLPModels
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using JSOSolvers
using SolverBenchmark
using Quadmath

FP = [Float16,Float32,Float64] # MPR2 Floating Point formats
HP_format = Float128 # High precision format
# evaluation error models
omega_obj = HP_format.((eps.((FP))))
omega_grad = HP_format.((eps.((FP))))
omega_obj[end] = HP_format(0)
omega_grad[end] = HP_format(0)

# values for mu decrease factor r
mu_factor = HP_format.([1,0.1,0.01])

# R2/MPR2 parameters
η0 = 0.05
η1=0.1
η2 =0.7
κm = 0.2
γ1 = 1/2
γ2 = 2.0
max_iter = 50000
atol = eps(FP[end])^(1/4)
rtol = atol

param = MPR2Params(HP_format.([η0, η1, η2, κm])...,Float16(γ1),Float16(γ2))

# status record structure
algos = [:R2,[Meta.parse("MPR2_mf_"*replace(string(Float64(mu_factor[i])),"." => "_")) for i in eachindex(mu_factor)]...]
names_str = ""
for algo in algos
  names_str *= ":"*string(algo)*","
end
df_str = ":name => String[], :status => Symbol[]"
col_str = [":neval_obj_",":neval_obj_fail_",":neval_grad_",":neval_grad_fail_"]
for col in col_str
  for fp in FP
    df_str *= ","*col*"$fp => Int64[]"
  end
end
names = eval(Meta.parse("[$names_str]"))
stats = Dict(name => eval(Meta.parse("DataFrame($df_str)")) for name in names)

mpmodel = nothing

stop_condition(m::FPMPNLPModel,solver::MPR2Solver,tol) = solver.g_norm <= tol # stopping condition for MPR2 (ignores euclidean norm computation error): is the same used by R2 for fair comparison

meta = OptimizationProblems.meta
names_pb_vars = meta[(meta.has_bounds .== false) .& (meta.ncon .== 0), [:nvar, :name]] #select unconstrained problems

pb_skip = ["vibrbeam"] # overflow in cos() argument of obj function causes error
for pb in eachrow(names_pb_vars)
  nlp = eval(Meta.parse("ADNLPProblems.$(pb[:name])(type=$(FP[end]), backend = :generic)"))
  @show nlp.meta.name
  if nlp.meta.name in pb_skip
    continue
  end
  try
    mpmodel = FPMPNLPModel(nlp,FP,HPFormat = HP_format,ωfRelErr = omega_obj,ωgRelErr = omega_grad); # instanciation checks might error in some case due to over/underflow with low-prec/low-range formats
  catch e
    continue
  end
  statr2 = R2(nlp,max_eval=max_iter,η1=η1,η2 = η2, γ1 = γ1,γ2 = γ2, atol = atol, rtol = rtol)
  push!(stats[:R2],(nlp.meta.name,
        statr2.status,
        [0 for _ in 1:(length(FP)-1)]...,
        nlp.counters.neval_obj,
        [0 for _ in 1:length(FP)]...,
        [0 for _ in 1:(length(FP)-1)]...,
        nlp.counters.neval_grad,
        [0 for _ in 1:length(FP)]...)
        )
  for i in eachindex(mu_factor)
    MultiPrecisionR2.reset!(mpmodel)
    statmpr2 = MPR2(mpmodel,max_iter = max_iter,run_free = true,par = param,mu_factor = mu_factor[i],stop_condition = stop_condition, atol = HP_format(atol), rtol = HP_format(rtol))
    push!(stats[algos[i+1]],
      [nlp.meta.name,
      statmpr2.status,
      [mpmodel.counters.neval_obj[fp] for fp in FP]...,
      [mpmodel.counters_fail.neval_obj[fp] for fp in FP]...,
      [mpmodel.counters.neval_grad[fp]  for fp in FP]...,
      [mpmodel.counters_fail.neval_grad[fp] for fp in FP]...]
    )
  end
end

#objective evaluation: time cost function
obj_time_cost_str = "obj_time_cost(df) = (df.status .!= :first_order) * Inf + sum(["
for i in eachindex(FP)
  obj_time_cost_str *= "1/2^(length(FP)-$i) .* df.neval_obj_$(FP[i]),"
end
obj_time_cost_str *=" ])"
eval(Meta.parse(obj_time_cost_str))

#objective evaluation: energy cost function
obj_nrg_cost_str = "obj_nrg_cost(df) = (df.status .!= :first_order) * Inf + sum(["
for i in eachindex(FP)
  obj_nrg_cost_str *= "1/4^(length(FP)-$i) .* df.neval_obj_$(FP[i]),"
end
obj_nrg_cost_str *=" ])"
eval(Meta.parse(obj_nrg_cost_str))

#gradient evaluation: time cost function
grad_time_cost_str = "grad_time_cost(df) = (df.status .!= :first_order) * Inf + sum(["
for i in eachindex(FP)
  grad_time_cost_str *= "1/2^(length(FP)-$i) .* df.neval_grad_$(FP[i]),"
end
grad_time_cost_str *=" ])"
eval(Meta.parse(grad_time_cost_str))

#gradient evaluation: energy cost function
grad_nrg_cost_str = "grad_nrg_cost(df) = (df.status .!= :first_order) * Inf + sum(["
for i in eachindex(FP)
  grad_nrg_cost_str *= "1/4^(length(FP)-$i) .* df.neval_grad_$(FP[i]),"
end
grad_nrg_cost_str *=" ])"
eval(Meta.parse(grad_nrg_cost_str))

#plot performance profiles
costs = [obj_time_cost, obj_nrg_cost, grad_time_cost, grad_nrg_cost]
costs_names = ["objective time", "objective energy", "gradient time", "gradient energy"]
legend = ["R2",["r-MPR2:a=$(Float64(mf))" for mf in mu_factor]...]
p_ot = performance_profile(stats,obj_time_cost,title = "Objective time effort")
p_oe = performance_profile(stats,obj_nrg_cost,title = "Objective energy effort")
p_gt = performance_profile(stats,grad_time_cost,title = "Gradient time effort")
p_ge = performance_profile(stats,grad_nrg_cost,title = "Gradient energy effort")

data_header_table = ["Algo","mu_fct",vcat([["eval_$fp","success_rate"] for fp in FP]...)..., "time_rat","nrg_rat","# solved(/$(nrow(stats[:R2])))"]

pb_solved = [nrow(filter(row -> row.status == :first_order,stats[algo])) for algo in algos] 

# obj eval stats : compute time and energy savings of MPR2 compare with R2
obj_eval_fields = [Meta.parse("neval_obj_$(FP[i])") for i in eachindex(FP)]
obj_eval_fail_fields = [Meta.parse("neval_obj_fail_$(FP[i])") for i in eachindex(FP)]

obj_eval = [[sum(stats[algos[i]][!,obj_eval_fields[j]]) for j in eachindex(FP)] for i in eachindex(algos)]
obj_eval_fail = [[sum(stats[algos[i]][!,obj_eval_fail_fields[j]]) for j in eachindex(FP)] for i in eachindex(algos)]
obj_total_eval = [sum([sum(stats[algos[i]][!,obj_eval_fields[j]]) for j in eachindex(FP)]) for i in eachindex(algos)]

obj_eval_ratio = [[obj_eval[i][j]/sum(obj_eval[i]) for i in eachindex(algos)] for j in eachindex(FP)]
obj_success_ratio = [[(obj_eval[i][j] - obj_eval_fail[i][j])/obj_eval[i][j] for i in eachindex(algos)] for j in eachindex(FP)]
obj_time_effort = sum.([obj_eval[j].*reverse([1.0/2^(i-1) for i in eachindex(FP)]) for j in eachindex(algos)])
obj_time_ratio = obj_time_effort./obj_time_effort[1]
obj_energy_effort = sum.([obj_eval[j].*reverse([1.0/4^(i-1) for i in eachindex(FP)]) for j in eachindex(algos)])
obj_energy_ratio = obj_energy_effort./obj_energy_effort[1]

obj_data = [[:R2,[:MPR2 for _ in eachindex(mu_factor)]...],[NaN,Float64.(mu_factor)...],vcat([[obj_eval_ratio[:][j],obj_success_ratio[:][j]] for j in eachindex(FP)]...)..., obj_time_ratio, obj_energy_ratio,pb_solved]

# grad eval stats : compute time and energy savings of MPR2 compare with R2
grad_eval_fields = [Meta.parse("neval_grad_$(FP[i])") for i in eachindex(FP)]
grad_eval_fail_fields = [Meta.parse("neval_grad_fail_$(FP[i])") for i in eachindex(FP)]

grad_eval = [[sum(stats[algos[i]][!,grad_eval_fields[j]]) for j in eachindex(FP)] for i in eachindex(algos)]
grad_eval_fail = [[sum(stats[algos[i]][!,grad_eval_fail_fields[j]]) for j in eachindex(FP)] for i in eachindex(algos)]
grad_total_eval = [sum([sum(stats[algos[i]][!,grad_eval_fields[j]]) for j in eachindex(FP)]) for i in eachindex(algos)]

grad_eval_ratio = [[grad_eval[i][j]/sum(grad_eval[i]) for i in eachindex(algos)] for j in eachindex(FP)]
grad_success_ratio = [[(grad_eval[i][j] - grad_eval_fail[i][j])/grad_eval[i][j] for i in eachindex(algos)] for j in eachindex(FP)]
grad_time_effort = sum.([grad_eval[j].*reverse([1.0/2^(i-1) for i in eachindex(FP)]) for j in eachindex(algos)])
grad_time_ratio = grad_time_effort./grad_time_effort[1]
grad_energy_effort = sum.([grad_eval[j].*reverse([1.0/4^(i-1) for i in eachindex(FP)]) for j in eachindex(algos)])
grad_energy_ratio = grad_energy_effort./grad_energy_effort[1]

grad_data = [[:R2,[:MPR2 for _ in eachindex(mu_factor)]...],[NaN,Float64.(mu_factor)...],vcat([[grad_eval_ratio[:][j],grad_success_ratio[:][j]] for j in eachindex(FP)]...)..., grad_time_ratio, grad_energy_ratio,pb_solved]

println("Objective evaluation stats:")
table_mpr2_obj = pretty_table(hcat(obj_data...),header = data_header_table)
println("Gradient evaluation stats:")
table_mpr2_grad = pretty_table(hcat(grad_data...),header = data_header_table)

# export paper's figure

```