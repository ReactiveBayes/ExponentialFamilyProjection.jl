@testitem "`MLEStrategy` should fail if given a function instead of a list of sampls" begin 
  using ExponentialFamily

  prj = ProjectedTo(Beta; parameters = ProjectionParameters(
    strategy = ExponentialFamilyProjection.MLEStrategy()
  ))

  @test_throws "`MLEStrategy` requires the projection argument to be an array of samples." project_to(prj, (x) -> 1)

end