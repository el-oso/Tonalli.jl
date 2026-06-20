# Fine-tuning orchestration. The honest v0.1 path: LoRA fine-tune on the iGPU (ROCm) or
# CPU, then deploy the adapter for NPU inference. The actual training loop lives in
# `ext/TonalliFineTuneExt.jl` (loaded when `PythonCall` is present) because it drives a
# ROCm PyTorch + transformers/peft stack; this file defines the config and entry point.

using TOML

"""
    LoRAConfig(; base_model, dataset, kwargs...)

Configuration for a LoRA fine-tune. `dataset` is a path to a JSONL file of
`{"text": ...}` or `{"messages": [...]}` records. `target_device` ∈ `("rocm", "cpu")`.
"""
struct LoRAConfig
    base_model::String
    dataset::String
    output_dir::String
    rank::Int
    alpha::Int
    dropout::Float64
    epochs::Int
    learning_rate::Float64
    max_seq_len::Int
    target_device::String
end
function LoRAConfig(;
        base_model::AbstractString,
        dataset::AbstractString,
        output_dir::AbstractString = "tonalli_lora_out",
        rank::Integer = 16,
        alpha::Integer = 32,
        dropout::Real = 0.05,
        epochs::Integer = 1,
        learning_rate::Real = 2.0e-4,
        max_seq_len::Integer = 2048,
        target_device::AbstractString = "rocm",
    )
    return LoRAConfig(
        String(base_model), String(dataset), String(output_dir),
        Int(rank), Int(alpha), Float64(dropout), Int(epochs),
        Float64(learning_rate), Int(max_seq_len), String(target_device),
    )
end

"""Load a [`LoRAConfig`](@ref) from a TOML file with a `[lora]` table."""
function LoRAConfig(path::AbstractString)
    t = TOML.parsefile(path)
    l = get(t, "lora", t)
    kw = Dict(Symbol(k) => v for (k, v) in l)
    return LoRAConfig(; kw...)
end

"""
    ROCmLoRATuner(config::LoRAConfig)

LoRA fine-tuner targeting the local AMD iGPU via ROCm (or CPU). Calling [`finetune`](@ref)
requires the `PythonCall` extension to be loaded.
"""
struct ROCmLoRATuner <: AbstractFineTuner
    config::LoRAConfig
end
ROCmLoRATuner(; kw...) = ROCmLoRATuner(LoRAConfig(; kw...))

"""
    finetune(t::ROCmLoRATuner) -> String

Run the LoRA fine-tune and return the path to the produced adapter directory. Requires
`using PythonCall` (loads `TonalliFineTuneExt`) and a ROCm PyTorch + transformers + peft + trl
environment.
"""
finetune(t::ROCmLoRATuner) = _finetune_impl(t)

# Overridden by TonalliFineTuneExt when PythonCall is loaded.
function _finetune_impl(::ROCmLoRATuner)
    return error(
        """
        Fine-tuning requires the PythonCall extension. Enable it with:

            julia> using PythonCall, Tonalli

        and provide a Python environment with ROCm PyTorch, `transformers`, `peft`, `trl`,
        and `datasets` installed. See the Tonalli docs "Fine-tuning" guide.
        """
    )
end
