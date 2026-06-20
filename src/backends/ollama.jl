# Ollama / llama.cpp backend — broad GGUF model coverage over CPU/Vulkan/ROCm. No true
# NPU acceleration, so this is the portable fallback. Talks to Ollama's OpenAI-compatible
# `/v1` surface; `pull_model` shells out to the `ollama` CLI when present.

"""
    OllamaBackend(model=""; host="127.0.0.1", port=11434)

Client for a running Ollama server (OpenAI-compatible endpoint). The portable fallback
backend — no NPU acceleration.
"""
struct OllamaBackend <: AbstractInferenceBackend
    model::String
    client::OpenAIClient
end
function OllamaBackend(model::AbstractString = ""; host::AbstractString = "127.0.0.1", port::Integer = 11434, timeout::Integer = 600)
    return OllamaBackend(String(model), OpenAIClient("http://$host:$(port)/v1"; timeout = timeout))
end

_ollama_model(b::OllamaBackend, model) = isempty(String(model)) ? b.model : String(model)

function chat(b::OllamaBackend, messages::AbstractVector; model::AbstractString = "", stream::Bool = false, on_token = nothing, kw...)
    m = _ollama_model(b, model)
    j = chat_completion(b.client, m, to_messages(messages); stream = stream, on_token = on_token, kw...)
    return parse_chat_response(j, m)
end
chat(b::OllamaBackend, messages::AbstractString; kw...) = chat(b, to_messages(messages); kw...)
complete(b::OllamaBackend, prompt::AbstractString; kw...) = chat(b, to_messages(prompt); kw...)

function list_models(b::OllamaBackend)
    out = ModelInfo[]
    j = try
        models(b.client)
    catch
        return out
    end
    for d in get(j, :data, [])
        push!(out, ModelInfo(String(get(d, :id, "?")), true, "", "", "", 0, 0.0))
    end
    return out
end

function health(b::OllamaBackend)
    up = ping(b.client)
    return HealthReport(up, [CheckResult("ollama server", up, up ? b.client.base_url : "unreachable", up ? "" : "Start Ollama (ollama serve).")])
end

function pull_model(::OllamaBackend, tag::AbstractString)
    ollama = Sys.which("ollama")
    ollama === nothing && error("ollama CLI not found on PATH")
    return success(pipeline(ignorestatus(`$ollama pull $tag`); stdout = stdout, stderr = stderr))
end
