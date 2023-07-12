@testset "Basic MPCounters check" begin
  nlp = SimpleNLPModel()
  T = [Float16, Float32]
  mpnlp = FPMPNLPModel(nlp, T)

  d = Dict([t => 0 for t in T])

  for counter in fieldnames(Counters)
    @eval @test $counter($nlp) == d
  end

  obj(mpnlp, mpnlp.meta.x0)
  grad(mpnlp, mpnlp.meta.x0)
  @test sum_counters(nlp) == 2

  for counter in fieldnames(Counters)
    increment!(nlp, counter, Float16)
  end
  # sum all counters of problem `mpnlp` except 
  # `cons`, `jac`, `jprod` and `jtprod` = 20-4+2
  d[Float16] = 18
  @test sum_counters(nlp) == d

  reset!(nlp)
  @test sum_counters(nlp) == Dict([t => 0 for t in T])

  for counter in fieldnames(Counters)
    increment!(nlp, counter, Float32)
    decrement!(nlp, counter, Float32)
  end
  @test sum_counters(nlp) == Dict([t => 0 for t in T])
end
