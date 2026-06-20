# Interface contracts (TypeContracts.jl). The `@contract` macro auto-defines the abstract
# type and any unqualified generic functions, so `chat`, `complete`, `health`, … become
# callable names here and backends add methods to them.
#
# We do NOT `@verify` in the module body: `check_contract` leans on `Base.return_types`,
# which is world-age-fragile during the defining module's own precompile (same lesson as
# Mexicah). Structural verification lives post-load in `test/contracts_test.jl`.

# `AbstractInferenceBackend`: an LLM inference runtime Tonalli can drive (e.g. FastFlowLM
# on the AMD NPU). Mandatory methods below; embed/serve!/stop! are optional and may `error`
# if unsupported. Docs are attached by the @contract macro from the description string.
@contract AbstractInferenceBackend "An LLM inference runtime Tonalli can drive." begin
    chat(::Self, ::AbstractVector)::ChatResponse => "run a (multi-turn) chat completion"
    complete(::Self, ::AbstractString)::ChatResponse => "single-prompt completion"
    list_models(::Self)::Vector => "models known to the backend"
    pull_model(::Self, ::AbstractString) => "download a model by tag/id"
    health(::Self)::HealthReport => "backend + hardware readiness"
    :optional
    embed(::Self, ::Any) => "embedding vector(s) for the input"
    serve!(::Self) => "start the backend's server process"
    stop!(::Self) => "stop the backend's server process"
end

# `AbstractModelSource`: a resolvable source of model weights — a HF repo, a local GGUF
# file, or a FastFlowLM tag. `resolve` returns a local path, downloading if necessary.
@contract AbstractModelSource "A resolvable source of model weights." begin
    resolve(::Self)::String => "resolve to a local path, downloading if needed"
    :optional
    metadata(::Self) => "model metadata, if available"
end

# `AbstractFineTuner`: a fine-tuning strategy. `finetune` runs the job and returns an
# adapter artifact (e.g. a path to a LoRA adapter directory).
@contract AbstractFineTuner "A fine-tuning strategy producing an adapter artifact." begin
    finetune(::Self)::String => "run fine-tuning, returning the adapter artifact path"
end
