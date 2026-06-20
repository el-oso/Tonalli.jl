
# API reference {#API-reference}

## Interfaces {#Interfaces}
<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.AbstractInferenceBackend' href='#Tonalli.AbstractInferenceBackend'><span class="jlbinding">Tonalli.AbstractInferenceBackend</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**TypeContracts Interface**

An LLM inference runtime Tonalli can drive.

**Mandatory methods**
- `chat(::Self, ::AbstractVector) :: ChatResponse` — run a (multi-turn) chat completion
  
- `complete(::Self, ::AbstractString) :: ChatResponse` — single-prompt completion
  
- `list_models(::Self) :: Vector` — models known to the backend
  
- `pull_model(::Self, ::AbstractString)` — download a model by tag/id
  
- `health(::Self) :: HealthReport` — backend + hardware readiness
  

**Optional methods**
- `embed(::Self, ::Any)` — embedding vector(s) for the input
  
- `serve!(::Self)` — start the backend's server process
  
- `stop!(::Self)` — stop the backend's server process
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/JuliaLang/julia/blob/15346901f0039751c5488744f1f62de7d87510a8/base/#L0" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.AbstractModelSource' href='#Tonalli.AbstractModelSource'><span class="jlbinding">Tonalli.AbstractModelSource</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**TypeContracts Interface**

A resolvable source of model weights.

**Mandatory methods**
- `resolve(::Self) :: String` — resolve to a local path, downloading if needed
  

**Optional methods**
- `metadata(::Self)` — model metadata, if available
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/JuliaLang/julia/blob/15346901f0039751c5488744f1f62de7d87510a8/base/#L0" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.AbstractFineTuner' href='#Tonalli.AbstractFineTuner'><span class="jlbinding">Tonalli.AbstractFineTuner</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



**TypeContracts Interface**

A fine-tuning strategy producing an adapter artifact.

**Mandatory methods**
- `finetune(::Self) :: String` — run fine-tuning, returning the adapter artifact path
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/JuliaLang/julia/blob/15346901f0039751c5488744f1f62de7d87510a8/base/#L0" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Backends {#Backends}
<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.FastFlowLM' href='#Tonalli.FastFlowLM'><span class="jlbinding">Tonalli.FastFlowLM</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FastFlowLM(model=""; host, port, pmode, binary)
```


A handle to a FastFlowLM runtime. Construct it, optionally [`serve!`](/api#Tonalli.serve!) a model, then [`chat`](/api#Tonalli.chat)/[`complete`](/api#Tonalli.complete). If a `flm serve` is already running, just point at its `host`/`port` and skip `serve!`.

`pmode` ∈ `("powersaver", "balanced", "performance", "turbo")`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L14-L22" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.LemonadeBackend' href='#Tonalli.LemonadeBackend'><span class="jlbinding">Tonalli.LemonadeBackend</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LemonadeBackend(model=""; host="127.0.0.1", port=8000)
```


Client for a running AMD Lemonade server. Inference works against an existing server; `pull_model`/`serve!` are not yet implemented.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/lemonade.jl#L5-L10" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.OllamaBackend' href='#Tonalli.OllamaBackend'><span class="jlbinding">Tonalli.OllamaBackend</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
OllamaBackend(model=""; host="127.0.0.1", port=11434)
```


Client for a running Ollama server (OpenAI-compatible endpoint). The portable fallback backend — no NPU acceleration.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/ollama.jl#L5-L10" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Inference verbs {#Inference-verbs}
<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.chat' href='#Tonalli.chat'><span class="jlbinding">Tonalli.chat</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
chat(b::FastFlowLM, messages; model="", temperature, max_tokens, stream, on_token, kw...)
```


Run a chat completion. `messages` may be a `String`, a `ChatMessage`, or a vector of `ChatMessage`/`Pair`/`Dict` (see `to_messages`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L49-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.complete' href='#Tonalli.complete'><span class="jlbinding">Tonalli.complete</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Single-prompt completion (wraps the prompt as one user turn).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L70" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.embed' href='#Tonalli.embed'><span class="jlbinding">Tonalli.embed</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
embed(b::FastFlowLM, input; model="")
```


Embedding vector(s) for `input` (a `String` or vector of `String`). Requires the server to have been started with embedding support (`serve!(b; embed=true)` or `flm serve … --embed 1`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L75-L80" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.list_models' href='#Tonalli.list_models'><span class="jlbinding">Tonalli.list_models</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



List models known to FastFlowLM (parsed from `flm list --json`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L90" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.pull_model' href='#Tonalli.pull_model'><span class="jlbinding">Tonalli.pull_model</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Download a model by FastFlowLM tag (e.g. `"llama3.2:1b"`) via `flm pull`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L116" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.serve!' href='#Tonalli.serve!'><span class="jlbinding">Tonalli.serve!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
serve!(b::FastFlowLM; model="", embed=false, wait=true, timeout=120)
```


Launch `flm serve` as a managed background process bound to `b.host`/`b.port`. When `wait=true`, blocks until the server answers `/models` (up to `timeout` seconds). Returns `b`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L127-L132" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.stop!' href='#Tonalli.stop!'><span class="jlbinding">Tonalli.stop!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Stop the managed `flm serve` process, if any.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L154" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.health' href='#Tonalli.health'><span class="jlbinding">Tonalli.health</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



Backend + hardware readiness — delegates to [`tonalli_doctor`](/api#Tonalli.tonalli_doctor).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/backends/fastflowlm.jl#L122" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Diagnostics {#Diagnostics}
<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.tonalli_doctor' href='#Tonalli.tonalli_doctor'><span class="jlbinding">Tonalli.tonalli_doctor</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
tonalli_doctor(; show = true) -> HealthReport
```


Probe the AMD NPU + iGPU stack and return a [`HealthReport`](/api#Tonalli.HealthReport). Checks the `flm` binary, the `amdxdna` driver and `/dev/accel` device, FastFlowLM's own NPU validation (`flm validate`), and — advisory, for fine-tuning — ROCm/iGPU availability.

`HealthReport.ready` reflects inference readiness on the NPU. Pass `show = false` to suppress printing.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/npu.jl#L35-L44" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.print_report' href='#Tonalli.print_report'><span class="jlbinding">Tonalli.print_report</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
print_report(r::HealthReport; io = stdout)
```


Pretty-print a [`HealthReport`](/api#Tonalli.HealthReport) with pass/fail markers and remediation advice.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/npu.jl#L139-L143" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.HealthReport' href='#Tonalli.HealthReport'><span class="jlbinding">Tonalli.HealthReport</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
HealthReport(ready, checks)
```


Aggregate diagnostic report. `ready` is the overall go/no-go for inference; `checks` carries the individual [`CheckResult`](/api#Tonalli.CheckResult)s. Render with [`print_report`](/api#Tonalli.print_report).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/types.jl#L64-L69" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.CheckResult' href='#Tonalli.CheckResult'><span class="jlbinding">Tonalli.CheckResult</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



One diagnostic probe: a named check with pass/fail, detail, and remediation advice.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/types.jl#L56" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Hub {#Hub}
<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.hf_download' href='#Tonalli.hf_download'><span class="jlbinding">Tonalli.hf_download</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
hf_download(repo_id, filename; revision="main", token=nothing) -> String
```


Download a single file from a Hugging Face repo and return its local cached path.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/hub.jl#L8-L12" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.gguf_metadata' href='#Tonalli.gguf_metadata'><span class="jlbinding">Tonalli.gguf_metadata</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
gguf_metadata(path) -> Dict{String,Any}
```


Read the key/value metadata block from a GGUF file (e.g. `general.architecture`, `*.context_length`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/hub.jl#L35-L40" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.HFModel' href='#Tonalli.HFModel'><span class="jlbinding">Tonalli.HFModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
HFModel(repo_id; filename="", revision="main")
```


A Hugging Face model source. With `filename` set, `resolve` downloads that file.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/hub.jl#L17-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Fine-tuning {#Fine-tuning}
<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.LoRAConfig' href='#Tonalli.LoRAConfig'><span class="jlbinding">Tonalli.LoRAConfig</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LoRAConfig(; base_model, dataset, kwargs...)
```


Configuration for a LoRA fine-tune. `dataset` is a path to a JSONL file of `{"text": ...}` or `{"messages": [...]}` records. `target_device` ∈ `("rocm", "cpu")`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/finetune.jl#L8-L13" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.ROCmLoRATuner' href='#Tonalli.ROCmLoRATuner'><span class="jlbinding">Tonalli.ROCmLoRATuner</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ROCmLoRATuner(config::LoRAConfig)
```


LoRA fine-tuner targeting the local AMD iGPU via ROCm (or CPU). Calling [`finetune`](/api#Tonalli.finetune) requires the `PythonCall` extension to be loaded.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/finetune.jl#L53-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.finetune' href='#Tonalli.finetune'><span class="jlbinding">Tonalli.finetune</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
finetune(t::ROCmLoRATuner) -> String
```


Run the LoRA fine-tune and return the path to the produced adapter directory. Requires `using PythonCall` (loads `TonalliFineTuneExt`) and a ROCm PyTorch + transformers + peft + trl environment.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/finetune.jl#L64-L70" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Value types {#Value-types}
<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.ChatMessage' href='#Tonalli.ChatMessage'><span class="jlbinding">Tonalli.ChatMessage</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ChatMessage(role, content)
```


A role-tagged chat turn. `role` is one of `"system"`, `"user"`, `"assistant"`, `"tool"`.

Convenience constructors accept a `Pair`, so `ChatMessage(:user => "hi")` works.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/types.jl#L4-L10" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.ChatResponse' href='#Tonalli.ChatResponse'><span class="jlbinding">Tonalli.ChatResponse</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ChatResponse
```


Result of a single completion. `raw` holds the backend's parsed JSON for callers that need fields Tonalli does not model.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/types.jl#L24-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.ModelInfo' href='#Tonalli.ModelInfo'><span class="jlbinding">Tonalli.ModelInfo</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ModelInfo
```


Metadata about a model known to a backend (mirrors FastFlowLM's `list --json`).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/types.jl#L41-L45" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Tonalli.Usage' href='#Tonalli.Usage'><span class="jlbinding">Tonalli.Usage</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



Token accounting returned by a backend, when available.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/el-oso/Tonalli.jl/blob/1c7876523716471bf66f3d174cf5d7711bedc56e/src/types.jl#L17" target="_blank" rel="noreferrer">source</a></Badge>

</details>

