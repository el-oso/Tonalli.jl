# Inference

Every backend implements [`AbstractInferenceBackend`](@ref): [`chat`](@ref),
[`complete`](@ref), [`list_models`](@ref), [`pull_model`](@ref), [`health`](@ref), and the
optional [`embed`](@ref), [`serve!`](@ref), [`stop!`](@ref). Write your code once; switch
runtimes by switching the constructor.

## FastFlowLM (NPU)

```julia
using Tonalli
b = FastFlowLM("llama3.2:1b"; pmode = "performance")   # powersaver|balanced|performance|turbo
serve!(b)
```

### Chatting

`messages` accepts a `String`, a `ChatMessage`, or a vector of `ChatMessage` / `Pair` /
`Dict`:

```julia
chat(b, "What is an NPU?")
chat(b, [
    ChatMessage("system", "You are terse."),
    :user => "Define XDNA.",
])
```

### Streaming

```julia
chat(b, "Write a haiku about silicon."; stream = true, on_token = print)
```

### Embeddings

Start the server with embedding support, then call [`embed`](@ref):

```julia
serve!(b; embed = true)
v = embed(b, "vectorize me")     # Vector{Vector{Float64}}
```

### Managing models

```julia
list_models(b)                    # ModelInfo[] (installed + available)
pull_model(b, "qwen3:4b")
```

## Pointing at an already-running server

If `flm serve` is already up (or behind a proxy), skip `serve!` and just construct the
handle with the right `host`/`port`:

```julia
b = FastFlowLM("llama3.2:1b"; host = "127.0.0.1", port = 52625)
chat(b, "hi")
```

## Other backends

```julia
LemonadeBackend("Llama-3.2-1B"; port = 8000)   # AMD hybrid NPU+iGPU (ONNX/VitisAI)
OllamaBackend("llama3.2"; port = 11434)        # portable GGUF fallback (CPU/Vulkan/ROCm)
```

These share the OpenAI-compatible client, so `chat`/`complete`/`embed` work against a
running server today; managed launch and pulls are being filled in (see [Roadmap](/roadmap)).
