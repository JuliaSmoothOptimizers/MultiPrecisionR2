# skip_int = false # waiting to know how to evaluate obj(Vector{AbstractFloat}) with IntervalArithmetic properly. see github issue https://github.com/JuliaIntervals/IntervalArithmetic.jl/issues/546

@testset "MPR2Precisions copy" begin
  p = MPR2Precisions(1)
  q = copy(p)
  @test [getfield(p, f) == getfield(q, f) for f in fieldnames(MPR2Precisions)] == [true for _ in fieldnames(MPR2Precisions)]
  p = MPR2Precisions(2)
  update_struct!(p, q)
  @test [getfield(p, f) == getfield(q, f) for f in fieldnames(MPR2Precisions)] == [true for _ in fieldnames(MPR2Precisions)]
  p = MPR2Precisions(2)
end
@testset "MPR2 Parameters " begin
  T = Float64
  L = Float16
  p = MPR2Params(T, L)

  for η0 in [-1.0, 10.0]
    p.η₀ = η0
    try
      MultiPrecisionR2.CheckMPR2ParamConditions(p)
      @test false
    catch e
      buf = IOBuffer()
      showerror(buf, e)
      err_msg = String(take!(buf))
      @test err_msg == "Expected 0 ≤ η₀ ≤ 1/2*η₁"
    end
  end

  p = MPR2Params(T, L)
  for η2 in [p.η₁ / 2, 10.0]
    p.η₂ = η2
    try
      MultiPrecisionR2.CheckMPR2ParamConditions(p)
      @test false
    catch e
      buf = IOBuffer()
      showerror(buf, e)
      err_msg = String(take!(buf))
      @test err_msg == "Expected η₁ ≤ η₂ < 1"
    end
  end

  p = MPR2Params(T, L)
  p.κₘ = 1.0
  try
    MultiPrecisionR2.CheckMPR2ParamConditions(p)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf, e)
    err_msg = String(take!(buf))
    @test err_msg == "Expected η₀+κₘ/2 ≤0.5*(1-η₂)"
  end

  p = MPR2Params(T, L)
  for γ1 in [p.γ₂ * 2, -1.0]
    p.γ₁ = γ1
    try
      MultiPrecisionR2.CheckMPR2ParamConditions(p)
      @test false
    catch e
      buf = IOBuffer()
      showerror(buf, e)
      err_msg = String(take!(buf))
      @test err_msg == "Expected 0<γ₁<1<γ₂"
    end
  end
end

@testset verbose = true "solve!" begin
  @testset "First eval overflow" begin
    Format = Float16
    f3(x) = prevfloat(typemax(Format)) * x[1]^2
    HPFormat = Format # set HPFormat to Float16 to force gradient error evaluation to overflow
    ωfRelErr = HPFormat[2.0]
    ωgRelErr = HPFormat[2.0]
    x0 = Format.([2])
    mpmodel =
      FPMPNLPModel(f3, x0, [Format], HPFormat = HPFormat, ωfRelErr = ωfRelErr, ωgRelErr = ωgRelErr)
    # objective function evaluation overflow
    stat = MPR2(mpmodel)
    @test stat.status == :exception
    @test stat.iter == 0
    @test stat.solution == Float16.(x0)
    @test stat.objective === Float16(Inf)
    # objective function error overflow
    x0 = Float16.([1.0])
    stat = MPR2(mpmodel, x₀ = x0)
    @test stat.status == :exception
    @test stat.iter == 0
    @test stat.solution == x0
    @test stat.objective === prevfloat(typemax(Float16))
    # gradient overflow
    x0 = Float16.([1.0])
    mpmodel =
      FPMPNLPModel(f3, x0, [Format], HPFormat = HPFormat, ωfRelErr = ωfRelErr, ωgRelErr = ωgRelErr)
    stat = MPR2(mpmodel, x₀ = x0)
    @test stat.status == :exception
    @test stat.iter == 0
    @test stat.solution == x0
    @test stat.dual_feas == Float16(Inf)
  end
end
@testset "computeStep!" begin
  @testset "underflow/overflow" begin
    FP = [Float16, Float32]
    gval = nextfloat(Float16(0.0)) * ones(2)
    g = (Float16.(gval), Float32.(gval))
    sval = ones(2)
    s = (Float16.(sval), Float32.(sval))
    σ = 2.0
    π = MPR2Precisions(1)
    # step underflow at Float16
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # step overflow at Float16
    gval = prevfloat(typemax(Float16)) * ones(2)
    g = (Float16.(gval), Float32.(gval))
    σ = 0.5
    π.πs = 1
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # step underflow at Float32
    π.πs = 1
    gval = nextfloat(Float16(0.0)) * ones(2)
    g = (Float16.(gval), Float32.(gval))
    σ = 2 * nextfloat(Float16(0.0)) / nextfloat(Float32(0.0))
    @test MultiPrecisionR2.computeStep!(s, g, σ, FP, π) == false
    @test π.πs == 2
    # σ underflow at Float16
    gval = Float32.(ones(2))
    g = (Float16.(gval), Float32.(gval))
    σ = Float32(nextfloat(Float16(0.0))) / 2.0
    π.πs = 1
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # sigma overflow at Float16
    σ = Float32(prevfloat(typemax(Float16))) * 2.0
    π.πs = 1
    MultiPrecisionR2.computeStep!(s, g, σ, FP, π)
    @test π.πs == 2
    @test s[π.πs] == -g[π.πs] / Float32(σ)
    # σ underflow at Float32
    gval = ones(2)
    g = (Float16.(gval), Float32.(gval))
    σ = nextfloat(Float32(0.0)) / 2.0
    π.πs = 1
    @test MultiPrecisionR2.computeStep!(s, g, σ, FP, π) == false
    @test π.πs == 2
    # sigma overflow at Float32
    σ = prevfloat(typemax(Float32)) * 2.0
    π.πs = 1
    @test MultiPrecisionR2.computeStep!(s, g, σ, FP, π) == false
    @test π.πs == 2
  end
end
@testset "computeCandidate!" begin
  @testset "over/underflow" begin
    FP = [Float16, Float32]
    cval = ones(2)
    c = (Float16.(cval), Float32.(cval))
    π = MPR2Precisions(1)
    # overflow at Float16
    xval = prevfloat(typemax(Float16)) * ones(2)
    x = (Float16.(xval), Float32.(xval))
    s = (Float16.(xval), Float32.(xval))
    MultiPrecisionR2.computeCandidate!(c, x, s, FP, π)
    @test π.πc == 2
    @test c[π.πc] == x[π.πc] .+ s[π.πc]
    # underflow at Float16
    xval = ones(2)
    x = (Float16.(xval), Float32.(xval))
    sval = eps(Float16) / 2.0 * ones(2)
    s = (Float16.(sval), Float32.(sval))
    MultiPrecisionR2.computeCandidate!(c, x, s, FP, π)
    @test π.πc == 2
    @test c[π.πc] == x[π.πc] .+ s[π.πc]
    #overflow at Float32
    xval = prevfloat(typemax(Float32)) * ones(2)
    x = (Float16.(xval), Float32.(xval))
    s = (Float16.(xval), Float32.(xval))
    @test MultiPrecisionR2.computeCandidate!(c, x, s, FP, π) == false
    @test π.πc == 2
    # underflow at Float32
    xval = ones(2)
    x = (Float16.(xval), Float32.(xval))
    sval = eps(Float32) / 2.0 * ones(2)
    s = (Float16.(sval), Float32.(sval))
    @test MultiPrecisionR2.computeCandidate!(c, x, s, FP, π) == false
    @test π.πc == 2
  end
end
@testset "computeModelDecrease!" begin
  FP = [Float16, Float32]
  π = MPR2Precisions(1)
  #overflow at Float16
  gval = Float32.(prevfloat(typemax(Float16)) * ones(2))
  g = (Float16.(gval), Float32.(gval))
  sval = ones(2) * 2.0
  s = (Float16.(sval), Float32.(sval))
  f(x) = sum(x)
  solver =
    MPR2Solver(FPMPNLPModel(f, similar(gval), FP, ωfRelErr = [0.0, 0.0], ωgRelErr = [0.0, 0.0]))
  MultiPrecisionR2.computeModelDecrease!(g, s, solver, FP, π)
  @test π.πΔ == 2
  @test solver.ΔT == -dot(g[π.πΔ], s[π.πΔ])
  #overflow at Float32
  π.πΔ = 1
  gval = prevfloat(typemax(Float32)) * ones(2)
  g = (Float16.(gval), Float32.(gval))
  @test MultiPrecisionR2.computeModelDecrease!(g, s, solver, FP, π) == false
  @test π.πΔ == 2
  @test solver.ΔT == -Inf
  # inconsistent FP format
  π = MPR2Precisions(1)
  π.πg = 2 # πg =2 πΔ = 1
  try
    MultiPrecisionR2.computeModelDecrease!(g, s, solver, FP, π)
    @test false
  catch e
    buf = IOBuffer()
    showerror(buf, e)
    err_msg = String(take!(buf))
    @test err_msg ==
          "Model decrease computation FP format should be greater that FP format of g and s"
  end
end
@testset "Mu related functions" begin
  f(x) = sum(x .^ 2)
  Formats = [Float32, Float64]
  dim = 2^5
  x0 = 1 / 10 .* ones(dim)
  gamma0(n, u) = 0.0
  m = FPMPNLPModel(f, x0, Formats, γfunc = gamma0)
  solver = MPR2Solver(m)
  solver.ϕ = 0.0
  u = eps(Float64) / 2
  @test MultiPrecisionR2.computeMu(m, solver) == (u + u) / (1 - u)
  m = FPMPNLPModel(f, x0, Formats)
  solver = MPR2Solver(m)
  π = MPR2Precisions(1)
  πr = copy(π)
  update_struct!(solver.π, π)
  mu = MultiPrecisionR2.computeMu(m, solver)
  mu_next = mu

  @test false == MultiPrecisionR2.recomputeMuPrecSelection!(solver.π, πr, solver.πmax)
  @test πr.πΔ == 2
  mu_next = MultiPrecisionR2.computeMu(m, solver; π = πr)
  @test mu_next < mu
  mu_next = mu
  update_struct!(solver.π, πr)

  @test false == MultiPrecisionR2.recomputeMuPrecSelection!(solver.π, πr, solver.πmax)
  @test πr.πnx == 2
  @test πr.πns == 2
  mu_next = MultiPrecisionR2.computeMu(m, solver; π = πr)
  @test mu_next < mu
  mu_next = mu
  update_struct!(solver.π, πr)

  @test false == MultiPrecisionR2.recomputeMuPrecSelection!(solver.π, πr, solver.πmax)
  @test πr.πs == 2
  mu_next = MultiPrecisionR2.computeMu(m, solver; π = πr)
  @test mu_next < mu
  mu_next = mu
  update_struct!(solver.π, πr)

  @test false == MultiPrecisionR2.recomputeMuPrecSelection!(solver.π, πr, solver.πmax)
  @test πr.πc == 2
  mu_next = MultiPrecisionR2.computeMu(m, solver; π = πr)
  @test mu_next < mu
  mu_next = mu
  update_struct!(solver.π, πr)

  @test false == MultiPrecisionR2.recomputeMuPrecSelection!(solver.π, πr, solver.πmax)
  @test πr.πg == 2
  mu_next = MultiPrecisionR2.computeMu(m, solver; π = πr)
  @test mu_next < mu
  update_struct!(solver.π, πr)

  @test true == MultiPrecisionR2.recomputeMuPrecSelection!(solver.π, πr, solver.πmax)
end

@testset "Default Callback" begin
  T = Float64
  f(x) = sum(x)
  Formats = [Float16, Float32]
  x0 = ones(Float32, 2)
  η0 = 0.01 # default η0 value upon solver instanciation
  ω = [0.1 * η0, 0.01 * η0]
  gamma0(n, u) = 0.0
  m = FPMPNLPModel(f, x0, Formats, γfunc = gamma0, ωfRelErr = ω, ωgRelErr = ω)
  solver = MPR2Solver(m)

  @testset "selectPif!()" begin
    # float32 eval overflow predicted
    solver.f = 2 * T(prevfloat(typemax(Float32)))
    solver.ΔT = 0.0
    solver.π.πf⁺ = 1
    MultiPrecisionR2.selectPif!(m, solver, 0.0)
    @test solver.π.πf⁺ == 2

    # float16 predicted error ok simple
    solver.f = 0.0
    solver.ΔT = 0.0
    solver.π.πf⁺ = 1
    solver.π.πc = 1
    MultiPrecisionR2.selectPif!(m, solver, 1.0)
    @test solver.π.πf⁺ == 1

    #float16 predicted error too big
    solver.f = 1.0
    solver.ΔT = 0.0
    solver.π.πf⁺ = 1
    MultiPrecisionR2.selectPif!(m, solver, solver.f * ω[1] / 2)
    @test solver.π.πf⁺ == 2
  end

  @testset "selectPic!()" begin
    solver.π.πf⁺ = 2
    MultiPrecisionR2.selectPic_default!(solver)
    @test solver.π.πc == 1
    solver.π.πf⁺ = 1
    MultiPrecisionR2.selectPic_default!(solver)
    @test solver.π.πc == 1
  end

  @testset "compute_f_at_c_default!()" begin
    solver.c[1] .= ones(Float16, 2)
    solver.c[2] .= ones(Float32, 2)
    solver.f = 0.0 # predicted error at c = ωfRelErr * solver.ΔT
    stats = GenericExecutionStats(m)
    # obj Float16 eval ok
    solver.ΔT = 1.0 # ωfBound = solver.ΔT * solver.p.η₀, pred error with Float16 = 0.1*ωfBound
    @test MultiPrecisionR2.compute_f_at_c_default!(m, solver, stats, nothing) == true
    @test solver.f⁺ == 2.0
    @test solver.π.πf⁺ == 1
    @test solver.ωf⁺ == ω[1] * solver.f⁺
    # Float16 err too big, Float32 ok
    solver.f = 2 * solver.ΔT * (2 - 1 / m.ωfRelErr[1]) # pred error at Float16 = 2*ωfBound
    @test MultiPrecisionR2.compute_f_at_c_default!(m, solver, stats, nothing) == true
    @test solver.f⁺ == 2.0
    @test solver.π.πf⁺ == 2
    @test solver.ωf⁺ == ω[2] * solver.f⁺
    # obj eval error too big
    solver.ΔT = 0.0 # i.e. ωfBound = 0
    solver.f = 1.0 #i.e. non null predicted relative error
    @test MultiPrecisionR2.compute_f_at_c_default!(m, solver, stats, nothing) == false
    # obj overflow
    solver.c[2] .= ones(Float32, 2) * prevfloat(typemax(Float32))
    solver.π.πf⁺ = 2
    @test MultiPrecisionR2.compute_f_at_c_default!(m, solver, stats, nothing) == false
  end

  @testset "compute_f_at_x_default!()" begin
    solver.x[1] .= ones(Float16, 2)
    solver.x[2] .= ones(Float32, 2)
    stats = GenericExecutionStats(m)
    # init eval
    solver.init = true
    solver.f = 1.0
    solver.ΔT = 0.0 # error bound = 0, no reachable with f=1.0 
    @test MultiPrecisionR2.compute_f_at_x_default!(m, solver, stats, nothing) == true # no error bound for init eval
    @test solver.f == 2.0

    solver.init = false
    # obj Float16 eval ok
    solver.f = 0.0
    solver.ΔT = 1.0
    solver.π.πf = 1
    solver.ωf = 0.0
    @test MultiPrecisionR2.compute_f_at_x_default!(m, solver, stats, nothing) == true # do nothing since ωf < error already
    @test solver.f == 0.0
    @test solver.π.πf == 1
    @test solver.ωf == 0.0
    # Float16 err too big, Float32 ok
    solver.π.πf = 1
    solver.ωf = 2 * solver.ΔT * solver.p.η₀ # error too big 
    @test MultiPrecisionR2.compute_f_at_x_default!(m, solver, stats, nothing) == true # compute f(x) 
    @test solver.f == 2.0
    @test solver.π.πf == 2
    @test solver.ωf == ω[2] * solver.f
    # obj eval error too big and max prec, do nothing
    solver.ΔT = 0.0 # i.e. ωfBound = 0
    solver.ωf = 1.0
    solver.π.πf = 2
    solver.f = -1.0
    @test MultiPrecisionR2.compute_f_at_x_default!(m, solver, stats, nothing) == false
    @test solver.f == -1.0 # did nothing, obj not recomputed
    # obj eval error too big
    solver.ΔT = 0.0 # i.e. ωfBound = 0
    solver.ωf = 1.0
    solver.π.πf = 1
    solver.f = -1.0
    @test MultiPrecisionR2.compute_f_at_x_default!(m, solver, stats, nothing) == false
    @test solver.f == 2.0 # did recompute f(x) with Float32
    @test solver.ωf == 2.0 * m.ωfRelErr[2]
    # obj overflow
    solver.x[2] .= ones(Float32, 2) * prevfloat(typemax(Float32))
    solver.ΔT = 1.0
    solver.π.πf = 1
    solver.ωf = 1.0 # error too big, has to recompute
    @test MultiPrecisionR2.compute_f_at_x_default!(m, solver, stats, nothing) == false # error due to obj overflow
    @test isinf(solver.f)
    @test solver.π.πf == 2
  end

  # gradient tests
  fq(x) = sum(x .^ 2)
  m = FPMPNLPModel(fq, x0, Formats, γfunc = gamma0, ωfRelErr = ω, ωgRelErr = ω)
  solver = MPR2Solver(m)
  stats = GenericExecutionStats(m)
  @testset "compute_g_default!()" begin
    solver.x[1] .= ones(Float16, 2)
    solver.x[2] .= ones(Float32, 2)
    solver.c[1] .= -ones(Float16, 2)
    solver.c[2] .= -ones(Float32, 2)
    # init
    solver.π.πx = 1
    solver.init = true
    MultiPrecisionR2.compute_g_default!(m, solver, stats, nothing)
    @test solver.g[1] == [2.0, 2.0]
    @test solver.π.πg == 1
    # in main loop
    solver.π.πc = 1
    solver.init = false
    MultiPrecisionR2.compute_g_default!(m, solver, stats, nothing)
    @test solver.g[1] == [-2.0, -2.0]
    @test solver.π.πg == 1
  end

  ω = [1.0, 1.0]
  m = FPMPNLPModel(fq, x0, Formats, γfunc = gamma0, ωfRelErr = ω, ωgRelErr = ω)
  solver = MPR2Solver(m)
  stats = GenericExecutionStats(m)
  @testset "recompute_g_default!()" begin
    # small step case
    solver.x[2] .= ones(Float32, 2)
    solver.s[2] .= solver.x[2] .* m.UList[2] # step too small
    solver.π.πx = 2
    solver.π.πs = 2
    @test MultiPrecisionR2.recompute_g_default!(m, solver, stats, nothing) == (false, false)
    @test stats.status == :small_step
    # not enough precision (mu too big), gradient recomputed
    solver.x[1] .= ones(Float16, 2)
    solver.x[2] .= ones(Float32, 2)
    solver.s[1] .= ones(Float16, 2)
    solver.s[2] .= ones(Float32, 2)
    solver.π.πx = 1
    solver.π.πs = 1
    solver.π.πg = 1
    solver.ωg = 1.0
    @test MultiPrecisionR2.recompute_g_default!(m, solver, stats, nothing) == (true, false)
    @test solver.π.πg == 2
    @test solver.g[2] == [2.0, 2.0]
    # enough precision with Float16
    ω = [0.0, 0.0]
    m = FPMPNLPModel(fq, x0, Formats, γfunc = gamma0, ωfRelErr = ω, ωgRelErr = ω)
    solver = MPR2Solver(m)
    solver.x[1] .= zeros(Float16, 2)
    solver.x[2] .= zeros(Float32, 2)
    solver.s[1] .= ones(Float16, 2)
    solver.s[2] .= ones(Float32, 2)
    solver.g[1] .= ones(Float16, 2)
    solver.π.πx = 1
    solver.π.πg = 1
    solver.ωg = 0.0
    @test MultiPrecisionR2.recompute_g_default!(m, solver, stats, nothing) == (false, true)
    @test solver.π.πg == 1
    @test solver.g[1] == [1.0, 1.0] # solver.g not modified since not recomputed
    # not enough precision with Float16, Float32 ok
    ω = [1.0, 0.0]
    m = FPMPNLPModel(fq, x0, Formats, γfunc = gamma0, ωfRelErr = ω, ωgRelErr = ω)
    solver = MPR2Solver(m)
    stats = GenericExecutionStats(m)
    solver.x[1] .= ones(Float16, 2)
    solver.x[2] .= ones(Float32, 2)
    solver.s[1] .= ones(Float16, 2)
    solver.s[2] .= ones(Float32, 2)
    solver.g[2] .= ones(Float32, 2)
    solver.π.πx = 1
    solver.π.πs = 1
    solver.π.πg = 1
    solver.ωg = 1.0
    @test MultiPrecisionR2.recompute_g_default!(m, solver, stats, nothing) == (true, true)
    @test solver.π.πg == 2
    @test solver.g[2] == [2.0, 2.0]
  end
end

@testset "checkUnderOverflow functions" begin
  @testset "Step" begin
    #ok case
    s = 2 .* ones(10)
    g = ones(10)
    @test MultiPrecisionR2.CheckUnderOverflowStep(s, g) == false
    #underflow
    s[1] = 0
    @test MultiPrecisionR2.CheckUnderOverflowStep(s, g) == true
    #overflow
    s[1] = Inf
  end
  @testset "Candidate" begin
    #ok case
    s = ones(10)
    c = 2 .* ones(10)
    x = ones(10)
    @test MultiPrecisionR2.CheckUnderOverflowCandidate(c, x, s) == false
    # ok case with step containing 0
    c[1] = x[1]
    s[1] = 0.0
    @test MultiPrecisionR2.CheckUnderOverflowCandidate(c, x, s) == false
    #underflow
    s[1] = 1.0
    @test MultiPrecisionR2.CheckUnderOverflowCandidate(c, x, s) == true
    #overflow
    c[1] = Inf
    @test MultiPrecisionR2.CheckUnderOverflowCandidate(c, x, s) == true
    #overflow
    c[1] = -Inf
    @test MultiPrecisionR2.CheckUnderOverflowCandidate(c, x, s) == true
  end

  @testset "Model decrease" begin
    for d in [0, -Inf, Inf]
      @test MultiPrecisionR2.CheckUnderOverflowMD(d) == true
    end
  end
end

@testset "Minimal problem tests" begin
  FPFormats = [Float16,Float32,Float64]
  atol = 1e-6
  rtol = 1e-6
  problem_set = SolverTest. unconstrained_nlp_set(gradient_backend = ADNLPModels.GenericForwardDiffADGradient)
  @testset "Interval Evaluation" begin
    setrounding(Interval,:accurate)
    for nlp in problem_set
      mpnlp = FPMPNLPModel(nlp,FPFormats)
      stats = MPR2(mpnlp,max_iter = 1000000, max_time = 60.0)
      ng0 = rtol != 0 ? norm(grad(nlp, nlp.meta.x0)) : 0
      ϵ = atol + rtol * ng0
      primal, dual = kkt_checker(nlp, stats.solution)
      if stats.status != :exception
        @test all(abs.(dual) .< ϵ)
        @test all(abs.(primal) .< ϵ)
        @test stats.dual_feas < ϵ
        @test stats.status == :first_order
      end
    end
  end
  @testset "Relative Error" begin
    omega = Float64.([sqrt(eps(t)) for t in FPFormats])
    omega[end] = 0.0
    for nlp in problem_set
      mpnlp = FPMPNLPModel(nlp,FPFormats;ωfRelErr = omega, ωgRelErr = omega)
      stats = MPR2(mpnlp, max_iter = 1000000, max_time = 60.0)
      ng0 = rtol != 0 ? norm(grad(nlp, nlp.meta.x0)) : 0
      ϵ = atol + rtol * ng0
      primal, dual = kkt_checker(nlp, stats.solution)
      if stats.status != :exception
        @test all(abs.(dual) .< ϵ)
        @test all(abs.(primal) .< ϵ)
        @test stats.dual_feas < ϵ
        @test stats.status == :first_order
      end
    end
  end
end
