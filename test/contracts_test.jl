@testitem "contracts: backends satisfy AbstractInferenceBackend" begin
    using Tonalli: FastFlowLM, LemonadeBackend, OllamaBackend, AbstractInferenceBackend
    using TypeContracts: satisfies
    for B in (FastFlowLM, LemonadeBackend, OllamaBackend)
        res = satisfies(B, AbstractInferenceBackend)
        @test res.satisfied || error("$B missing: $(res.missing_methods)")
    end
end

@testitem "contracts: HFModel satisfies AbstractModelSource" begin
    using Tonalli: HFModel, AbstractModelSource
    using TypeContracts: satisfies
    res = satisfies(HFModel, AbstractModelSource)
    @test res.satisfied || error("HFModel missing: $(res.missing_methods)")
end

@testitem "contracts: CommandLineTuner satisfies AbstractFineTuner" begin
    using Tonalli: CommandLineTuner, AbstractFineTuner
    using TypeContracts: satisfies
    res = satisfies(CommandLineTuner, AbstractFineTuner)
    @test res.satisfied || error("CommandLineTuner missing: $(res.missing_methods)")
end
