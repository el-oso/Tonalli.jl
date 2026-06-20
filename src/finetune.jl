# Fine-tuning orchestration. Project policy: NO Python in any capacity — only .so libraries
# (via ccall) and command-line tools. So Tonalli does not embed a trainer; it drives an
# external command-line trainer through `CommandLineTuner`, and the long-term plan is a
# pure-Julia LoRA trainer (Enzyme/Optimisers on AMDGPU.jl) — see the docs roadmap.

using TOML

"""
    LoRAConfig(; base_model, dataset, kwargs...)

Configuration for a LoRA fine-tune. `dataset` is a path to a JSONL file of
`{"text": ...}` or `{"messages": [...]}` records. `target_device` ∈ `("rocm", "cpu")`.
Consumed by a [`CommandLineTuner`](@ref)'s command builder.
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
    CommandLineTuner(config; command)

Fine-tune by invoking an **external command-line trainer** — no Python, no in-process
runtime; just a CLI tool (or a `.so`-backed binary) that Tonalli runs as a subprocess.

`command` is a function `(::LoRAConfig) -> Cmd` that builds the trainer invocation from the
config. [`finetune`](@ref) runs it and returns the produced adapter directory
(`<output_dir>/adapter`).

Tonalli ships no bundled trainer (there is no mature non-Python LoRA CLI for AMD yet), so
you supply one — e.g. a future `flm finetune`, a llama.cpp finetune binary, or your own
tool. A native pure-Julia LoRA trainer is on the roadmap.

```julia
t = CommandLineTuner(cfg; command = c -> `my-trainer --model \$(c.base_model) --data \$(c.dataset) --out \$(c.output_dir) --rank \$(c.rank)`)
finetune(t)
```
"""
struct CommandLineTuner <: AbstractFineTuner
    config::LoRAConfig
    command::Function
end
CommandLineTuner(config::LoRAConfig; command::Function) = CommandLineTuner(config, command)

"""
    finetune(t::CommandLineTuner) -> String

Run the external trainer and return the path to the produced adapter directory.
Throws if the command is malformed or the trainer exits non-zero.
"""
function finetune(t::CommandLineTuner)
    cmd = t.command(t.config)
    cmd isa Cmd || throw(ArgumentError("CommandLineTuner.command must return a `Cmd`, got $(typeof(cmd))"))
    ok = success(pipeline(ignorestatus(cmd); stdout = stdout, stderr = stderr))
    ok || error("fine-tune command failed (non-zero exit): $cmd")
    return abspath(joinpath(t.config.output_dir, "adapter"))
end
