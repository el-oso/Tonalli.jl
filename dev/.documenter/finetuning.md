
# Fine-tuning {#Fine-tuning}

## The honest hardware story {#The-honest-hardware-story}

The XDNA NPU is, in practice, an **inference accelerator**. On-NPU _training_ is research only. AMD's actual supported workflow — and Tonalli's — is:
> 
> **LoRA fine-tune on the iGPU (ROCm) or CPU → deploy the adapter for NPU inference.**
> 


So fine-tuning runs on your **gfx115x iGPU** via ROCm PyTorch, and the resulting adapter is merged/converted for FastFlowLM to serve on the NPU.

## Requirements {#Requirements}

Fine-tuning lives in the `TonalliFineTuneExt` extension, loaded when `PythonCall` is present. You need a Python environment with a **ROCm build of PyTorch** plus `transformers`, `peft`, `trl`, and `datasets`.

```julia
using PythonCall, Tonalli
```


RDNA 3.5 iGPUs (gfx1150/1151/1152) often need `HSA_OVERRIDE_GFX_VERSION` set for ROCm. [`tonalli_doctor`](/api#Tonalli.tonalli_doctor) flags ROCm availability.

## Run a LoRA fine-tune {#Run-a-LoRA-fine-tune}

```julia
cfg = LoRAConfig(
    base_model    = "meta-llama/Llama-3.2-1B-Instruct",
    dataset       = "train.jsonl",     # {"text": ...} or {"messages": [...]} per line
    output_dir    = "out",
    rank          = 16,
    epochs        = 1,
    target_device = "rocm",            # or "cpu"
)
adapter = finetune(ROCmLoRATuner(cfg))
```


Or from a TOML file:

```toml
# tune.toml
[lora]
base_model = "meta-llama/Llama-3.2-1B-Instruct"
dataset = "train.jsonl"
rank = 16
epochs = 1
```


```bash
tonalli finetune tune.toml
```


`finetune` returns the path to the saved adapter directory (safetensors).

## Serving the adapted model on the NPU {#Serving-the-adapted-model-on-the-NPU}

After fine-tuning, merge the LoRA adapter into the base weights and convert to FastFlowLM's model format, then `serve!` as usual. (FastFlowLM adapter-at-serve support is evolving; the merge-and-convert path always works.) See [Roadmap](/roadmap) for the native-Julia training track that will remove the Python dependency entirely.
