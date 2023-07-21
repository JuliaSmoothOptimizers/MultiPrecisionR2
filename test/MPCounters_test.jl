@testset "Basic MPCounters check" begin
  T = [Float16, Float32]
  f(x) = x[1]
  nlp = ADNLPModel(f, Float32[1.0])
  mpnlp = FPMPNLPModel(nlp, T, ωfRelErr = [0.0, 0.0], ωgRelErr = [0.0, 0.0])

  for counter in fieldnames(MPCounters)
    @eval @test $counter($nlp) == 0
  end

  obj(mpnlp, mpnlp.meta.x0)
  grad(mpnlp, mpnlp.meta.x0)
  d = Dict([t => 0 for t in T])
  d[Float32] = 2
  @test MultiPrecisionR2.sum_counters(mpnlp) == d

  for counter in fieldnames(MPCounters)
    increment!(mpnlp, counter, Float16)
  end
  # sum all counters of problem `mpnlp` except 
  # `cons`, `jac`, `jprod` and `jtprod` = 20-4(Float16)+2(Float32)
  d[Float16] = 16
  @test MultiPrecisionR2.sum_counters(mpnlp) == d

  reset!(mpnlp)
  @test MultiPrecisionR2.sum_counters(mpnlp) == Dict([t => 0 for t in T])

  for counter in fieldnames(MPCounters)
    increment!(mpnlp, counter, Float32)
    decrement!(mpnlp, counter, Float32)
  end
  @test MultiPrecisionR2.sum_counters(mpnlp) == Dict([t => 0 for t in T])
end