using MultiPrecisionR2

using LinearAlgebra
using Logging
using Test
using ADNLPModels
using NLPModels
using IntervalArithmetic
using OptimizationProblems
using OptimizationProblems.ADNLPProblems
using JSOSolvers
using SolverCore
using SolverTest
using ManualNLPModels

with_logger(NullLogger()) do
  @testset "utils.jl" begin
    println("#### Testing utils...")
    t = @elapsed include("utils_test.jl")
    println("#### done (took $t seconds).")
  end
  @testset "MPCounters.jl" begin
    println("#### Testing MPCounters...")
    t = @elapsed include("MPCounters_test.jl")
    println("#### done (took $t seconds).")
  end
  @testset "MPNLPModels.jl" begin
    println("#### Testing MPNLPModels...")
    t = @elapsed include("MPNLPModels_test.jl")
    println("#### done (took $t seconds).")
  end
  @testset "MultiPrecisionR2.jl" begin
    println("#### Testing MultiPrecisionR2...")
    t = @elapsed include("MultiPrecisionR2_test.jl")
    println("#### done (took $t seconds).")
  end
  @testset "R2_equivalence_test.jl" begin
    println("#### Testing Equivalence with R2 from JSOSolvers.jl...")
    t = @elapsed include("R2_equivalence_test.jl")
    println("#### done (took $t seconds).")
  end
end
