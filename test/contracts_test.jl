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

@testitem "contracts: ROCmLoRATuner satisfies AbstractFineTuner" begin
    using Tonalli: ROCmLoRATuner, AbstractFineTuner
    using TypeContracts: satisfies
    res = satisfies(ROCmLoRATuner, AbstractFineTuner)
    @test res.satisfied || error("ROCmLoRATuner missing: $(res.missing_methods)")
end
