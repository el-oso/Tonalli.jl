"""
    Tonalli

A Julia toolkit for running and fine-tuning LLMs on AMD Ryzen AI hardware.

`v0.1` is an *integration layer*: it gives one unified, contract-checked Julia API over
best-in-class runtimes — [FastFlowLM](https://fastflowlm.com) for NPU (XDNA2) inference,
plus interfaces for AMD Lemonade and Ollama/llama.cpp — together with Hugging Face model
download (`HuggingFaceApi`), GGUF metadata (`GGUFFiles`), and a fine-tuning interface that
drives external command-line trainers. The project uses no Python in any capacity — only
`.so` libraries (via `ccall`) and command-line tools.

Start with [`tonalli_doctor`](@ref) to check your NPU/iGPU stack.
"""
module Tonalli

using TypeContracts: @contract, check_contract, satisfies, InterfaceError

include("types.jl")
include("contracts.jl")
include("npu.jl")
include("openai_client.jl")
include("hub.jl")
include("backends/fastflowlm.jl")
include("backends/lemonade.jl")
include("backends/ollama.jl")
include("finetune.jl")
include("cli.jl")

# Core data types
export ChatMessage, ChatResponse, ModelInfo, Usage, CheckResult, HealthReport

# Interfaces (generic functions defined by the @contract macro)
export AbstractInferenceBackend, AbstractModelSource, AbstractFineTuner
export chat, complete, embed, list_models, pull_model, health, serve!, stop!

# Backends
export FastFlowLM, LemonadeBackend, OllamaBackend

# Diagnostics
export tonalli_doctor, print_report

# Hub
export hf_download, gguf_metadata, HFModel

# Fine-tuning
export LoRAConfig, CommandLineTuner, finetune

end # module Tonalli
