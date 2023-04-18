using MultiPrecisionR2
using Test
using Logging

#include("MPNLPModels_test.jl")
#include("MultiPrecisionR2_test.jl")

# MPNLPModels_test()

with_logger(NullLogger()) do
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
end