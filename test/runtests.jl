using MultiPrecisionR2

using ForwardDiff
using LinearAlgebra
using Logging
using Test
using ADNLPModels
using IntervalArithmetic

with_logger(NullLogger()) do
  @testset "MultiPrecisionR2.jl" begin
    println("#### Testing MultiPrecisionR2...")
    t = @elapsed include("MultiPrecisionR2_test.jl")
    println("#### done (took $t seconds).")
  end
  @testset "MPNLPModels.jl" begin
    println("#### Testing MPNLPModels...")
    t = @elapsed include("MPNLPModels_test.jl")
    println("#### done (took $t seconds).")
  end
end