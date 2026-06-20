# Lemonade backend — AMD's hybrid NPU+iGPU server (ONNX Runtime + VitisAI), exposed over
# an OpenAI-compatible API. v0.1 talks to an already-running Lemonade server; managed
# launch and model pulls are stubbed for a future release.

"""
    LemonadeBackend(model=""; host="127.0.0.1", port=8000)

Client for a running AMD Lemonade server. Inference works against an existing server;
`pull_model`/`serve!` are not yet implemented.
"""
struct LemonadeBackend <: AbstractInferenceBackend
    model::String
    client::OpenAIClient
end
function LemonadeBackend(model::AbstractString = ""; host::AbstractString = "127.0.0.1", port::Integer = 8000, timeout::Integer = 600)
    return LemonadeBackend(String(model), OpenAIClient("http://$host:$(port)/api/v1"; timeout = timeout))
end

_lemon_model(b::LemonadeBackend, model) = isempty(String(model)) ? b.model : String(model)

function chat(b::LemonadeBackend, messages::AbstractVector; model::AbstractString = "", stream::Bool = false, on_token = nothing, kw...)
    m = _lemon_model(b, model)
    j = chat_completion(b.client, m, to_messages(messages); stream = stream, on_token = on_token, kw...)
    return parse_chat_response(j, m)
end
chat(b::LemonadeBackend, messages::AbstractString; kw...) = chat(b, to_messages(messages); kw...)
complete(b::LemonadeBackend, prompt::AbstractString; kw...) = chat(b, to_messages(prompt); kw...)

function list_models(b::LemonadeBackend)
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

function health(b::LemonadeBackend)
    up = ping(b.client)
    return HealthReport(up, [CheckResult("lemonade server", up, up ? b.client.base_url : "unreachable", up ? "" : "Start the Lemonade server (lemonade-server serve).")])
end

pull_model(::LemonadeBackend, ::AbstractString) =
    error("LemonadeBackend.pull_model is not implemented yet — pull via the Lemonade CLI for now.")
