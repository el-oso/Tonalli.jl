---
layout: home
hero:
  name: Tonalli.jl
  text: LLMs on AMD Ryzen AI
  tagline: Run and fine-tune large language models on the AMD NPU (XDNA2) and iGPU — from Julia.
  actions:
    - theme: brand
      text: Getting started
      link: /getting_started
    - theme: alt
      text: API reference
      link: /api
---


# Tonalli.jl {#Tonalli.jl}

The Julia ecosystem has great glue for _talking to_ LLM servers, but no first-class story for **running models on AMD Ryzen AI NPUs** or **fine-tuning** them locally. Tonalli fills that gap.

`v0.1` is an **integration layer**: one unified, contract-checked Julia API over best-in-class runtimes.

|            Capability |                                                                                       How |
| ---------------------:| -----------------------------------------------------------------------------------------:|
|     **NPU inference** |                           [FastFlowLM](https://fastflowlm.com) (XDNA2), OpenAI-compatible |
|   **Hybrid NPU+iGPU** |                                                          AMD Lemonade _(interface ready)_ |
| **Portable fallback** |                                                    Ollama / llama.cpp _(interface ready)_ |
|    **Model download** |                                                  Hugging Face Hub via `HuggingFaceApi.jl` |
|   **GGUF inspection** |                                                                            `GGUFFiles.jl` |
|       **Fine-tuning** |                                                LoRA on the iGPU (ROCm) → serve on the NPU |
|       **Diagnostics** | [`tonalli_doctor`](/api#Tonalli.tonalli_doctor) — one command to validate the whole stack |


All backends implement a single [`AbstractInferenceBackend`](/api#Tonalli.AbstractInferenceBackend) contract (via [TypeContracts.jl](https://github.com/el-oso)), so swapping runtimes never changes your code.

```julia
using Tonalli

tonalli_doctor()                       # check NPU/iGPU readiness

b = FastFlowLM("llama3.2:1b")
serve!(b)                              # start the NPU server
println(chat(b, "Explain XDNA in one sentence.").content)
```

> 
> The name _Tōnalli_ (Nahuatl) is the inner fire / animating spirit — a sibling to [Mexicah.jl](https://github.com/el-oso/Mexicah.jl).
> 

