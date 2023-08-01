# MPCounters

`MPCounters` structure is meant to extend the `Counters` structure from [`NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) package to multi-precision.
`MPCounters` fields are dictionaries `Dict{DataType,Int}`, counting evaluations for each FP formats.

# Fields

Fields are exactly the same than `Counters` structure, but are dictionaries instead of `Int`. See [Counters.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/main/src/nlp/counters.jl).

# Interface

`neval_field(mpnlp)`: returns the total number of any FP formats of `field` evaluations.

`neval_field(mpnlp,T)`: returns the number of `field` evaluations with FP format `T`.

`sum_eval(mpnlp)`: returns sum of all counters of any FP formats except `cons`, `jac`, `jprod`, `jtprod`. 
