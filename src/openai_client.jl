# Minimal OpenAI-compatible HTTP client, shared by every server backend (FastFlowLM,
# Lemonade, Ollama all expose an OpenAI `/v1` surface). Handles chat/completions,
# embeddings, model listing, and Server-Sent-Events streaming.

using HTTP
using JSON3

"""
    OpenAIClient(base_url; api_key, timeout)

Thin client for an OpenAI-compatible endpoint. `base_url` should include the version
prefix, e.g. `"http://127.0.0.1:52625/v1"`.
"""
struct OpenAIClient
    base_url::String
    api_key::String
    timeout::Int
end
function OpenAIClient(base_url::AbstractString; api_key::AbstractString = "sk-no-key", timeout::Int = 600)
    return OpenAIClient(rstrip(base_url, '/'), api_key, timeout)
end

_headers(c::OpenAIClient) = [
    "Content-Type" => "application/json",
    "Authorization" => "Bearer $(c.api_key)",
]

"""Dig a nested value out of a parsed-JSON object, returning `default` if any hop fails."""
function _dig(obj, path::Vararg{Union{Symbol, Int}}; default = nothing)
    cur = obj
    for k in path
        try
            if k isa Int
                (cur isa AbstractVector && 1 <= k <= length(cur)) || return default
                cur = cur[k]
            else
                cur = get(cur, k, nothing)
            end
            cur === nothing && return default
        catch
            return default
        end
    end
    return cur
end

function _post(c::OpenAIClient, path::AbstractString, body)
    url = string(c.base_url, path)
    resp = HTTP.post(url, _headers(c), JSON3.write(body); readtimeout = c.timeout, retry = false)
    return JSON3.read(resp.body)
end

"""Reachability probe: GET `/models`, returning `true` on HTTP 2xx."""
function ping(c::OpenAIClient)
    try
        resp = HTTP.get(string(c.base_url, "/models"), _headers(c); readtimeout = 5, retry = false, status_exception = false)
        return resp.status < 400
    catch
        return false
    end
end

"""
    chat_completion(c, model, messages; temperature, max_tokens, stream, on_token, extra...)

POST `/chat/completions`. `messages` is a vector of [`ChatMessage`](@ref). When
`stream = true`, `on_token` (a `String -> Any` callback) is invoked per delta token and
the concatenated text is returned as a synthetic non-streaming response object.
"""
function chat_completion(
        c::OpenAIClient, model::AbstractString, messages::AbstractVector{ChatMessage};
        temperature = nothing, max_tokens = nothing, stream::Bool = false,
        on_token = nothing, extra...,
    )
    body = Dict{String, Any}(
        "model" => model,
        "messages" => [Dict("role" => m.role, "content" => m.content) for m in messages],
    )
    temperature === nothing || (body["temperature"] = temperature)
    max_tokens === nothing || (body["max_tokens"] = max_tokens)
    for (k, v) in extra
        body[string(k)] = v
    end
    stream && return _chat_stream(c, model, body, on_token)
    return _post(c, "/chat/completions", body)
end

function _chat_stream(c::OpenAIClient, model::AbstractString, body::Dict, on_token)
    body["stream"] = true
    url = string(c.base_url, "/chat/completions")
    buf = IOBuffer()
    finish = "stop"
    # Read raw bytes as they arrive (readavailable, not readline — chunked SSE frames don't
    # line up with HTTP chunk boundaries) and split into "\n\n"-delimited events ourselves.
    pending = ""
    consume(event) = for line in split(event, '\n')
        startswith(line, "data:") || continue
        data = strip(line[6:end])
        (isempty(data) || data == "[DONE]") && continue
        local j
        try
            j = JSON3.read(data)
        catch
            continue
        end
        tok = _dig(j, :choices, 1, :delta, :content; default = "")
        if tok isa AbstractString && !isempty(tok)
            print(buf, tok)
            on_token === nothing || on_token(tok)
        end
        fr = _dig(j, :choices, 1, :finish_reason)
        fr isa AbstractString && (finish = fr)
    end
    HTTP.open("POST", url, _headers(c); readtimeout = c.timeout, retry = false) do io
        write(io, JSON3.write(body))
        HTTP.closewrite(io)
        HTTP.startread(io)
        while !eof(io)
            pending *= String(readavailable(io))
            while (idx = findfirst("\n\n", pending)) !== nothing
                consume(pending[1:(first(idx) - 1)])
                pending = pending[(last(idx) + 1):end]
            end
        end
        isempty(pending) || consume(pending)
    end
    text = String(take!(buf))
    # Synthesize an OpenAI-shaped object so callers parse streamed + non-streamed uniformly.
    # Symbol keys to match `_dig`/`parse_chat_response` (which use the JSON3 Symbol access).
    return Dict{Symbol, Any}(
        :model => model,
        :choices => [Dict{Symbol, Any}(:message => Dict{Symbol, Any}(:role => "assistant", :content => text), :finish_reason => finish)],
    )
end

"""POST `/embeddings`; returns the parsed response object."""
function embeddings(c::OpenAIClient, model::AbstractString, input)
    return _post(c, "/embeddings", Dict("model" => model, "input" => input))
end

"""GET `/models`; returns the parsed response object."""
function models(c::OpenAIClient)
    resp = HTTP.get(string(c.base_url, "/models"), _headers(c); readtimeout = c.timeout, retry = false)
    return JSON3.read(resp.body)
end

# ── Shared parsing helpers (used by every backend) ──────────────────────────

"""Build a [`ChatResponse`](@ref) from an OpenAI-shaped (possibly `Dict`) chat object."""
function parse_chat_response(j, fallback_model::AbstractString)
    content = _dig(j, :choices, 1, :message, :content; default = "")
    content === nothing && (content = "")
    finish = _dig(j, :choices, 1, :finish_reason; default = "stop")
    usage = nothing
    if _dig(j, :usage) !== nothing
        usage = Usage(
            Int(_dig(j, :usage, :prompt_tokens; default = 0)),
            Int(_dig(j, :usage, :completion_tokens; default = 0)),
            Int(_dig(j, :usage, :total_tokens; default = 0)),
        )
    end
    return ChatResponse(
        String(content),
        String(_dig(j, :model; default = fallback_model)),
        String(finish === nothing ? "stop" : finish),
        usage,
        j,
    )
end

"""Normalize loose user input into a `Vector{ChatMessage}`.

Accepts a `String` (→ single user turn), a single `ChatMessage`, or a vector of
`ChatMessage` / `Pair` (`:user => "hi"`) / `AbstractDict` (`Dict("role"=>..,"content"=>..)`).
"""
to_messages(s::AbstractString) = [ChatMessage("user", s)]
to_messages(m::ChatMessage) = [m]
to_messages(v::AbstractVector{ChatMessage}) = v
function to_messages(v::AbstractVector)
    return ChatMessage[_as_message(x) for x in v]
end
_as_message(m::ChatMessage) = m
_as_message(p::Pair) = ChatMessage(p)
function _as_message(d::AbstractDict)
    role = haskey(d, "role") ? d["role"] : d[:role]
    content = haskey(d, "content") ? d["content"] : d[:content]
    return ChatMessage(String(role), String(content))
end
