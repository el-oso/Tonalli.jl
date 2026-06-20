# Fine-tuning

## Policy: no Python

Tonalli uses **no Python in any capacity** — only `.so` libraries (via `ccall`) and
command-line tools. The Python ML stack (PyTorch / transformers / peft / trl) is therefore
**out of scope**. Fine-tuning is exposed as an interface that drives an *external
command-line trainer*, with a native pure-Julia trainer planned (see [Roadmap](/roadmap)).

## The hardware story

The XDNA NPU is, in practice, an **inference accelerator** — training runs on the iGPU
(ROCm) or CPU, and the resulting adapter is merged/converted for NPU inference. Tonalli
does not bundle a trainer (there is no mature non-Python LoRA CLI for AMD yet), so you point
it at one.

## `CommandLineTuner`

`CommandLineTuner` runs an external CLI trainer as a subprocess and returns the produced
adapter directory. You provide a `command` builder, `(::LoRAConfig) -> Cmd`:

```julia
using Tonalli

cfg = LoRAConfig(
    base_model = "meta-llama/Llama-3.2-1B-Instruct",
    dataset    = "train.jsonl",     # {"text": ...} or {"messages": [...]} per line
    output_dir = "out",
    rank       = 16,
    epochs     = 1,
)

tuner = CommandLineTuner(cfg; command = c -> `my-trainer
    --model $(c.base_model) --data $(c.dataset)
    --out $(c.output_dir) --rank $(c.rank) --epochs $(c.epochs)`)

adapter = finetune(tuner)   # returns "<output_dir>/adapter"
```

`finetune` throws if the command is malformed or the trainer exits non-zero.

## From the CLI

`tonalli finetune <config.toml>` reads a `trainer` argv from the `[lora]` table and
substitutes config placeholders:

```toml
[lora]
base_model = "meta-llama/Llama-3.2-1B-Instruct"
dataset = "train.jsonl"
output_dir = "out"
rank = 16
epochs = 1
trainer = ["my-trainer", "--model", "{base_model}", "--data", "{dataset}",
           "--out", "{output_dir}", "--rank", "{rank}", "--epochs", "{epochs}"]
```

Placeholders: `{base_model} {dataset} {output_dir} {rank} {alpha} {epochs} {lr}
{max_seq_len} {device}`. The argv is run directly (no shell), so paths with spaces are safe.

```bash
tonalli finetune tune.toml
```

## Serving the adapted model on the NPU

After training, merge the LoRA adapter into the base weights and convert to FastFlowLM's
model format, then [`serve!`](@ref) as usual. (FastFlowLM adapter-at-serve support is
evolving; merge-and-convert always works.) See [Roadmap](/roadmap) for the native-Julia
LoRA trainer that will remove the need for an external tool entirely.
