
# Roadmap {#Roadmap}

## v0.1 — integration layer (current) {#v0.1-—-integration-layer-current}
- ✅ FastFlowLM NPU backend (chat, complete, embed, streaming, model mgmt, lifecycle)
  
- ✅ `tonalli_doctor` stack diagnostics
  
- ✅ Hugging Face download + GGUF metadata
  
- ✅ LoRA fine-tuning on the iGPU (ROCm) via the PythonCall extension
  
- ✅ `AbstractInferenceBackend` contract + `tonalli` CLI
  
- 🚧 Lemonade & Ollama backends: managed launch + `pull_model`
  
- 🚧 Adapter → NPU-serve bridge (merge + convert; track FastFlowLM adapter support)
  

## Future: a pure-Julia engine {#Future:-a-pure-Julia-engine}

The long game is to replace the wrapped runtimes with a native engine behind the **same** contracts (no API churn for users):
- **Weights** — read GGUF (`GGUFFiles.jl`) + safetensors; reuse `Transformers.jl` model definitions where available.
  
- **Inference** — the transformer forward pass as `KernelAbstractions.jl` kernels on `AMDGPU.jl` (gfx1152 iGPU first, CPU fallback). NPU acceleration via the MLIR-AIE / IRON / XRT toolchain is a later, research-grade step.
  
- **Native fine-tuning** — LoRA with `Enzyme`/`Zygote` + `Optimisers.jl` on the iGPU, removing the Python dependency entirely. This is the real "fill the gap" payoff.
  

When it lands, it slots in as another `AbstractInferenceBackend` / `AbstractFineTuner`.
