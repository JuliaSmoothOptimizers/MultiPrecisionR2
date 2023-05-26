@testset "Equivalence with R2 with one precision (tests similar dual_feas, solution, iterations)" begin
  T = Float64
  omega = T.([0.0]) # no relative error
  γfunc(n,u) = T(0.0) # no dot product error
  HPFormat = T # forces MPR2 to behave as R2
  η1 = T(0.02)
  η2 = T(0.95)
  γ1 = T(1/2)
  γ2 = 1/γ1
  nvar = 100
  max_iter = 1000
  max_time = Inf
  σmin = T(0)
  mpr2param = MPR2Params(T,T)
  mpr2param.γ₁ = γ1
  mpr2param.γ₂ = γ2
  mpr2param.η₁ = η1
  mpr2param.η₂ = η2
  problems = setdiff(names(OptimizationProblems.ADNLPProblems), [:ADNLPProblems])
  for s in problems
    nlp = eval(s)(type = Val(T),n = nvar);
    if nlp.meta.ncon == 0 && sum(isinf.(nlp.meta.uvar)) == nlp.meta.nvar && sum(isinf.(nlp.meta.lvar)) == nlp.meta.nvar # unconstrained problems
      nlpList = [nlp]
      mpmodel = FPMPNLPModel(nlpList,HPFormat=HPFormat,ωfRelErr=omega,ωgRelErr=omega,γfunc=γfunc);
      mpr2solver = MPR2Solver(mpmodel);
      @testset "Testing $(nlp.meta.name)" begin
        statr2 = R2(nlp,γ1 = γ1,η1 = η1, η2 = η2,max_time = max_time,σmin = σmin,max_eval = max_iter,verbose=1)
        statmpr2 = MultiPrecisionR2.solve!(mpr2solver,mpmodel,par = mpr2param,max_iter = max_iter,σmin = σmin,verbose=1)
        if (statmpr2.status != :exception # might happen that mpr2 stops if μ too big (μ ≠ 0 even if ωg = 0), or under/overflow
          && statmpr2.status != :small_step) # mpr2 returns small_step if relative step size is too small, r2 if absolute step size too small
          @test statr2.iter == statmpr2.iter
          @test statr2.dual_feas == statmpr2.dual_feas
          @test statr2.solution == statmpr2.solution
        end
      end
    end
  end
end

  