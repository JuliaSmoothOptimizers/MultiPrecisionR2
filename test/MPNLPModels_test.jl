# Try to run 2 times if some tests do not pass


@testset "Warnings and errors for FP formats ill-initialization" begin
  f(x) = x[1]+x[2]
  x0 = zeros(2)
  # Test FP format model order in MList
  Formats = [Float64,Float32]
  nlpList = [ADNLPModel(f,fp.(x0)) for fp in Formats]
  try
    #@test_logs (:error,"MList: wrong models FP formats precision order")
    FPMPNLPModel(nlpList)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "MList: wrong models FP formats precision order"
  end
  # Test HPFormat 
  reverse!(nlpList)
  @test_logs (:warn,"HPFormat (Float64) is the same format than highest accuracy NLPModel: chances of numerical instability increased") min_level=Logging.Warn FPMPNLPModel(nlpList)
  HPFormat = Float32
  try
    FPMPNLPModel(nlpList,HPFormat=HPFormat)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "HPFormat ($HPFormat) must be a FP format with precision equal or greater than NLPModels (max prec NLPModel: Float64)" 
  end
end

@testset "MList test mismatch nlps dimension initialization" begin
  f1(x) = x[1]+x[2]
  f2(x) = x[1]
  x0 = zeros(2)
  nlpList = [ADNLPModel(f1,x0),ADNLPModel(f2,[1.0])]
  try
    FPMPNLPModel(nlpList)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "FPMPModel.MList NLPModels dimensions mismatch"
  end
end

@testset "ωfRelErr and ωgRelErr mismatch dimensions" begin
  f3(x) = x[1]+x[2]
  x0 = zeros(2)
  nlpList = [ADNLPModel(f3,x0),ADNLPModel(f3,x0)]
  ωfRelErr = [0.0]
  try
    FPMPNLPModel(nlpList,ωfRelErr=ωfRelErr)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "MList and ωfRelErr dimension mismatch"
  end
  ωgRelErr = [0.0]
  try
    FPMPNLPModel(nlpList,ωgRelErr=ωgRelErr)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "MList and ωgRelErr dimension mismatch"
  end
end

@testset "γfunc callback test" begin
  f4(x) = x[1]+x[2]
  x0 = zeros(2)
  nlpList = [ADNLPModel(f4,x0)]
  γfunc = 0
  try
    FPMPNLPModel(nlpList,γfunc = γfunc)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf,e)
    err_msg = String(take!(buf))
    @test err_msg == "Wrong γfunc template, expected template: γfunc(n::Int,u::Float) -> Float"
  end
end

@testset "Default Interval instanciation" begin
  f5(x) = x[1]+x[2]
  x0 = zeros(2)
  Formats = [Float32,Float64]
  nlpList = [ADNLPModel(f5,fp.(x0)) for fp in Formats]
  @test_logs (:info,"Interval evaluation used by default for objective error evaluation: might significantly increase computation time")
  (:info,"Interval evaluation used by default for gradient error evaluation: might significantly increase computation time")
  MPnlp=FPMPNLPModel(nlpList)
  @test MPnlp.ωfRelErr == Vector{Float64}()
  @test MPnlp.ωgRelErr == Vector{Float64}()
end

@testset verbose = true "Obj and grad evaluation" begin
  @testset "Input FP formats consistency" begin
    setrounding(Interval,:accurate)
    f6(x) = x[1]+x[2]
    x0 = zeros(2)
    Formats = [Float32,Float64]
    nlpList = [ADNLPModel(f6,fp.(x0)) for fp in Formats]
    MPnlp=FPMPNLPModel(nlpList)
    funclist = [MultiPrecisionR2.objmp,MultiPrecisionR2.objerrmp,MultiPrecisionR2.gradmp,MultiPrecisionR2.graderrmp]
    for f in funclist
      try 
        f(MPnlp,x0,1)
        @test false
      catch e
        buf = IOBuffer()
        showerror(buf,e)
        err_msg = String(take!(buf))
        @test err_msg == "Expected input format Float32 for x but got Float64"
      end
    end
    try 
      MultiPrecisionR2.gradmp!(MPnlp,x0,1,copy(x0))
      @test false
    catch e
      buf = IOBuffer()
      showerror(buf,e)
      err_msg = String(take!(buf))
      @test err_msg == "Expected input format Float32 for x but got Float64"
    end
  end
  @testset "Outputs FP formats consistency" begin
    setrounding(Interval,:accurate)
    f7(x) = x[1]+x[2]
    x0 = zeros(2)
    Formats = [Float16,Float32,Float64]
    nlpList = [ADNLPModel(f7,fp.(x0)) for fp in Formats]
    MPnlp=FPMPNLPModel(nlpList)
    for i in 1:length(Formats)
      fp = Formats[i]
      @test typeof(MultiPrecisionR2.objmp(MPnlp,fp.(x0),i)) == Formats[i]
      @test typeof(MultiPrecisionR2.objerrmp(MPnlp,fp.(x0),i)[1]) == Formats[i]
      @test typeof(MultiPrecisionR2.gradmp(MPnlp,fp.(x0),i)) == Vector{Formats[i]}
      @test typeof(MultiPrecisionR2.graderrmp(MPnlp,fp.(x0),i)[1]) == Vector{Formats[i]}
      g = copy(fp.(x0))
      MultiPrecisionR2.gradmp!(MPnlp,fp.(x0),i,g)
      @test typeof(g) == Vector{Formats[i]}
    end
  end
  @testset "Overflow interval eval" begin
    Format = Float16
    setrounding(Interval,:accurate)
    c = prevfloat(typemax(Format))
    #obj test
    f8(x) = c*x[1]
    x0 = ones(2)
    nlpList = [ADNLPModel(f8,Format.(x0))]
    MPnlp=FPMPNLPModel(nlpList)
    @test MultiPrecisionR2.objerrmp(MPnlp,Format.(x0),1) == (0,Inf)
    #grad testf(x) = c+x[1]
    @test MultiPrecisionR2.graderrmp(MPnlp,Format.(x0),1) == (zeros(2),Inf)
  end
  @testset "Overflow floating point eval" begin
    Format = Float16
    setrounding(Interval,:accurate)
    c = prevfloat(typemax(Format))
    #obj test
    f9(x) = (10/eps(Format)+c)*x[1]
    x0 = ones(1)
    nlpList = [ADNLPModel(f9,Format.(x0))]
    ωfRelErr = [0.0]
    ωgRelErr = [0.0]
    MPnlp=FPMPNLPModel(nlpList, ωfRelErr = ωfRelErr, ωgRelErr = ωgRelErr)
    @test MultiPrecisionR2.objerrmp(MPnlp,Format.(x0),1) == (Inf,Inf)
    #grad testf(x) = c+x[1]
    @test MultiPrecisionR2.graderrmp(MPnlp,Format.(x0),1)[2] == Inf
  end
end