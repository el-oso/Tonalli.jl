
# Inference {#Inference}

Every backend implements [`AbstractInferenceBackend`](/api#Tonalli.AbstractInferenceBackend): [`chat`](/api#Tonalli.chat), [`complete`](/api#Tonalli.complete), [`list_models`](/api#Tonalli.list_models), [`pull_model`](/api#Tonalli.pull_model), [`health`](/api#Tonalli.health), and the optional [`embed`](/api#Tonalli.embed), [`serve!`](/api#Tonalli.serve!), [`stop!`](/api#Tonalli.stop!). Write your code once; switch runtimes by switching the constructor.

## FastFlowLM (NPU) {#FastFlowLM-NPU}

```julia
using Tonalli
b = FastFlowLM("llama3.2:1b"; pmode = "performance")   # powersaver|balanced|performance|turbo
serve!(b)
```


### Chatting {#Chatting}

`messages` accepts a `String`, a `ChatMessage`, or a vector of `ChatMessage` / `Pair` / `Dict`:

```julia
chat(b, "What is an NPU?")
chat(b, [
    ChatMessage("system", "You are terse."),
    :user => "Define XDNA.",
])
```


### Streaming {#Streaming}

```julia
chat(b, "Write a haiku about silicon."; stream = true, on_token = print)
```


### Embeddings {#Embeddings}

Start the server with embedding support, then call [`embed`](/api#Tonalli.embed):

```julia
serve!(b; embed = true)
v = embed(b, "vectorize me")     # Vector{Vector{Float64}}
```


### Managing models {#Managing-models}

```julia
list_models(b)                    # ModelInfo[] (installed + available)
pull_model(b, "qwen3:4b")
```


## Pointing at an already-running server {#Pointing-at-an-already-running-server}

If `flm serve` is already up (or behind a proxy), skip `serve!` and just construct the handle with the right `host`/`port`:

```julia
b = FastFlowLM("llama3.2:1b"; host = "127.0.0.1", port = 52625)
chat(b, "hi")
```


## Other backends {#Other-backends}

```julia
LemonadeBackend("Llama-3.2-1B"; port = 8000)   # AMD hybrid NPU+iGPU (ONNX/VitisAI)
OllamaBackend("llama3.2"; port = 11434)        # portable GGUF fallback (CPU/Vulkan/ROCm)
```


These share the OpenAI-compatible client, so `chat`/`complete`/`embed` work against a running server today; managed launch and pulls are being filled in (see [Roadmap](/roadmap)).
