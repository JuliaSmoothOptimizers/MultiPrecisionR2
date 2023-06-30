# Try to run 2 times if some tests do not pass

@testset "Warnings and errors for FP formats ill-initialization" begin
  f(x) = x[1]+x[2]
  x0 = zeros(2)
  # Test FP format model order in FPList
  Formats = [Float64,Float32]
  try
    FPMPNLPModel(f,x0,Formats)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "FList must be ordered by increasing precision order (e.g [Float16,Float32,Float64])"
  end
  # Test HPFormat 
  reverse!(Formats)
  @test_logs (:warn,"HPFormat (Float64) is the same format than highest accuracy NLPModel: chances of numerical instability increased") min_level=Logging.Warn FPMPNLPModel(f,x0,Formats)
  HPFormat = Float32
  try
    FPMPNLPModel(f,x0,Formats,HPFormat=HPFormat)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "HPFormat ($HPFormat) must be a FP format with precision equal or greater than NLPModels (max prec NLPModel: Float64)" 
  end
  type = Float16
  x0 = type.(x0)
  try
    FPMPNLPModel(f,x0,Formats)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "eltype of x0 ($type) must be in FPList ($Formats)" 
  end
end

@testset "ωfRelErr and ωgRelErr mismatch dimensions" begin
  Formats = [Float32,Float64]
  f3(x) = x[1]+x[2]
  x0 = zeros(2)
  ωfRelErr = [0.0]
  try
    FPMPNLPModel(f3,x0,Formats,ωfRelErr=ωfRelErr)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "DimensionError: Input ωfRelErr should have length 2 not 1"
  end
  ωgRelErr = [0.0]
  try
    FPMPNLPModel(f3,x0,Formats,ωgRelErr=ωgRelErr)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "DimensionError: Input ωgRelErr should have length 2 not 1"
  end
end

@testset "γfunc callback test" begin
  Formats = [Float64]
  f4(x) = x[1]+x[2]
  x0 = zeros(2)
  γfunc = 0
  try
    FPMPNLPModel(f4,x0,Formats,γfunc = γfunc)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "Wrong γfunc template, expected template: γfunc(n::Int,u::Float) -> Float"
  end
  Formats = [Float16]
  x0 = Float16.(zeros(Int32(round(3/eps(Float16)))))
  try
    FPMPNLPModel(f4,x0,Formats)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "γfunc: dot product error greater than 100% with highest precision. Consider using higher precision floating point format, or provide a different callback function for γfunc (last option might cause numerical instability)."
  end
end

@testset "Default Interval instanciation" begin
  f5(x) = x[1]+x[2]
  x0 = zeros(2)
  Formats = [Float32,Float64]
  @test_logs (:info,"Interval evaluation used by default for objective error evaluation: might significantly increase computation time")
  (:info,"Interval evaluation used by default for gradient error evaluation: might significantly increase computation time")
  MPnlp=FPMPNLPModel(f5,x0,Formats)
  @test MPnlp.ωfRelErr == Vector{Float64}()
  @test MPnlp.ωgRelErr == Vector{Float64}()
end

@testset verbose = true "Obj and grad evaluation" begin
  @testset "Input FP formats consistency" begin
    setrounding(Interval,:accurate)
    f6(x) = x[1]+x[2]
    x0 = zeros(2)
    Formats = [Float32,Float64]
    MPnlp=FPMPNLPModel(f6,x0,Formats)
    x16 = Float16.(x0)
    try 
      objerrmp(MPnlp,x16)
      @test false
    catch e
      buf = IOBuffer()
      showerror(buf,e)
      err_msg = String(take!(buf))
      @test err_msg == "Floating point format of x (Float16) not supported by the multiprecison model (FP formats supported: $(Formats))"
    end
    try 
      graderrmp(MPnlp,x16)
      @test false
    catch e
      buf = IOBuffer()
      showerror(buf,e)
      err_msg = String(take!(buf))
      @test err_msg == "Floating point format of x (Float16) not supported by the multiprecison model (FP formats supported: $(Formats))"
    end
  end
  @testset "Overflow interval eval" begin
    Format = Float16
    setrounding(Interval,:accurate)
    c = prevfloat(typemax(Format))
    #obj test
    f8(x) = c*x[1]
    x0 = ones(2)
    MPnlp=FPMPNLPModel(f8,Format.(x0),[Format])
    @test objerrmp(MPnlp,Format.(x0)) == (0,Inf)
    #grad testf(x) = c+x[1]
    @test graderrmp(MPnlp,Format.(x0)) == (zeros(2),Inf)
  end
  @testset "Overflow floating point eval" begin
    Format = Float16
    setrounding(Interval,:accurate)
    c = prevfloat(typemax(Format))
    #obj test
    f9(x) = (10/eps(Format)+c)*x[1]
    x0 = ones(1)
    ωfRelErr = [0.0]
    ωgRelErr = [0.0]
    MPnlp=FPMPNLPModel(f9,Format.(x0),[Format], ωfRelErr = ωfRelErr, ωgRelErr = ωgRelErr)
    @test objerrmp(MPnlp,Format.(x0)) == (Inf,Inf)
    #grad test f(x) = c+x[1]
    @test graderrmp(MPnlp,Format.(x0))[2] == Inf
  end
end