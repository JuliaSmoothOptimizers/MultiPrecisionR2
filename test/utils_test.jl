@testset "Overflows" begin
  @test MultiPrecisionR2.check_overflow(NaN) == true
  @test MultiPrecisionR2.check_overflow(Inf) == true
  @test MultiPrecisionR2.check_overflow(-Inf) == true
  @test MultiPrecisionR2.check_overflow(0.0) == false

  @test MultiPrecisionR2.check_overflow(0.0 .. Inf) == true
  @test MultiPrecisionR2.check_overflow(-Inf .. 0.0) == true
  @test MultiPrecisionR2.check_overflow(-Inf .. Inf) == true
  @test MultiPrecisionR2.check_overflow(0.0 .. 0.0) == false

  g = zeros(10)
  @test MultiPrecisionR2.check_overflow(g) == false
  g[1] = Inf
  @test MultiPrecisionR2.check_overflow(g) == true
  g[1] = -Inf
  @test MultiPrecisionR2.check_overflow(g) == true
  g[1] = NaN
  @test MultiPrecisionR2.check_overflow(g) == true

  G = [0.0 .. 0.0 for _ =1:10]
  @test MultiPrecisionR2.check_overflow(G) == false
  G[1] = 0.0 .. Inf
  @test MultiPrecisionR2.check_overflow(g) == true

end

@testset "Gamma function" begin
  f(x) = 0.0
  try
    MultiPrecisionR2.γfunc_test_template(f)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf, e)
    err_msg = String(take!(buf))
    @test err_msg == "Wrong γfunc template, expected template: γfunc(n::Int,u::Float) -> Float"
  end
  f(n,u) = 2.0
  try
    MultiPrecisionR2.γfunc_test_error_bound(0,0.0,f)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf, e)
    err_msg = String(take!(buf))
    @test err_msg == "γfunc: dot product error greater than 100% with highest precision. Consider using higher precision floating point format, or provide a different callback function for γfunc (last option might cause numerical instability)."
  end
end

@testset "Eval test functions" begin
  f(x) = 0.0
  FPList = [Float32]
  x0 = zeros(1)
  nlp = ADNLPModel(f,x0)
  try
    MultiPrecisionR2.ObjIntervalEval_test(nlp, FPList)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf, e)
    err_msg = String(take!(buf))
    @test err_msg == "Objective function evaluation error with interval, error model must be provided. Error detail:"
  end
  try
    MultiPrecisionR2.GradIntervalEval_test(nlp, FPList)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf, e)
    err_msg = String(take!(buf))
    @test err_msg == "Gradient evaluation error with interval, error model must be provided. Error detail:"
  end
  
end