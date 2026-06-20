# Tonalli.jl

[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/Tonalli.jl/dev/)
[![CI](https://github.com/el-oso/Tonalli.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/el-oso/Tonalli.jl/actions/workflows/CI.yml)
[![Docs build](https://github.com/el-oso/Tonalli.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/el-oso/Tonalli.jl/actions/workflows/Documentation.yml)

📖 **Documentation: <https://el-oso.github.io/Tonalli.jl/dev/>**

Run and fine-tune LLMs on **AMD Ryzen AI** hardware — the NPU (XDNA 2) and iGPU — from Julia.

The Julia ecosystem has clients for LLM servers but no first-class story for AMD NPU
inference or local fine-tuning. Tonalli fills that gap. `v0.1` is an **integration layer**:
one unified, contract-checked API over [FastFlowLM](https://fastflowlm.com) (NPU), with
interfaces ready for AMD Lemonade and Ollama/llama.cpp, plus Hugging Face download, GGUF
inspection, and a fine-tuning hook for external CLI trainers.

> **No Python.** Tonalli uses only `.so` libraries (via `ccall`) and command-line tools —
> never an embedded Python runtime.

```julia
using Tonalli

tonalli_doctor()                      # validate the NPU/iGPU stack

b = FastFlowLM("llama3.2:1b")
serve!(b)                             # start the NPU server
println(chat(b, "Explain XDNA in one sentence.").content)
stop!(b)
```

## Highlights

- **NPU inference** via FastFlowLM (OpenAI-compatible): `chat`, `complete`, `embed`,
  streaming, model management, managed server lifecycle.
- **One contract, many runtimes** — `AbstractInferenceBackend` (TypeContracts.jl); swap
  FastFlowLM / Lemonade / Ollama without changing your code.
- **`tonalli_doctor`** — one call validates driver, firmware, memlock, device, and ROCm.
- **Hugging Face + GGUF** — `hf_download`, `gguf_metadata`.
- **Fine-tuning** — `CommandLineTuner` drives an external CLI trainer (no bundled trainer;
  pure-Julia LoRA on the roadmap) → serve the adapter on the NPU.
- **`tonalli` CLI** — `doctor`, `pull`, `serve`, `chat`, `finetune`.

## Requirements

AMD Ryzen AI 300-series (or other XDNA 2) chip; Linux kernel 7.0+ with `amdxdna`; AMD XRT;
NPU firmware ≥ 1.1.0.0; FastFlowLM (`flm`) on `PATH`. For fine-tuning, an external
command-line trainer of your choice. Run `tonalli doctor` to check.

## Status

Early (`v0.1`). FastFlowLM backend is complete; Lemonade/Ollama managed launch and the
adapter→NPU-serve bridge are in progress. A pure-Julia inference/training engine is the
long-term goal — see the docs Roadmap.

## Tests

```bash
julia --project=test test/runtests.jl
```

Hardware tests are tagged `:npu` and skipped automatically when no `/dev/accel` device is
present; network tests (`:network`) run only with `TONALLI_TEST_NETWORK=1`.
