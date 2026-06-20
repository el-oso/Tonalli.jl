# FastFlowLM backend — NPU-native inference on AMD XDNA2 via the `flm` CLI + its
# OpenAI-compatible server. This is the reference, fully-implemented backend.

"""Default FastFlowLM server port (asks `flm port`, falls back to 52625)."""
function flm_default_port(binary::AbstractString = something(flm_binary(), "flm"))
    ok, out = _capture(`$binary port`)
    if ok
        m = match(r"(\d+)", out)
        m === nothing || return parse(Int, m.captures[1])
    end
    return 52625
end

"""
    FastFlowLM(model=""; host, port, pmode, binary)

A handle to a FastFlowLM runtime. Construct it, optionally [`serve!`](@ref) a model, then
[`chat`](@ref)/[`complete`](@ref). If a `flm serve` is already running, just point at its
`host`/`port` and skip `serve!`.

`pmode` ∈ `("powersaver", "balanced", "performance", "turbo")`.
"""
struct FastFlowLM <: AbstractInferenceBackend
    model::String
    host::String
    port::Int
    pmode::String
    binary::String
    client::OpenAIClient
    proc::Base.RefValue{Union{Base.Process, Nothing}}
end

function FastFlowLM(
        model::AbstractString = "";
        host::AbstractString = "127.0.0.1",
        port::Integer = flm_default_port(),
        pmode::AbstractString = "performance",
        binary::AbstractString = something(flm_binary(), "flm"),
        timeout::Integer = 600,
    )
    client = OpenAIClient("http://$host:$(port)/v1"; timeout = timeout)
    return FastFlowLM(String(model), String(host), Int(port), String(pmode), String(binary), client, Ref{Union{Base.Process, Nothing}}(nothing))
end

_model_or(b::FastFlowLM, model) = isempty(String(model)) ? b.model : String(model)

# ── Inference ───────────────────────────────────────────────────────────────

"""
    chat(b::FastFlowLM, messages; model="", temperature, max_tokens, stream, on_token, kw...)

Run a chat completion. `messages` may be a `String`, a `ChatMessage`, or a vector of
`ChatMessage`/`Pair`/`Dict` (see `to_messages`).
"""
function chat(
        b::FastFlowLM, messages::AbstractVector;
        model::AbstractString = "", temperature = nothing, max_tokens = nothing,
        stream::Bool = false, on_token = nothing, kw...,
    )
    m = _model_or(b, model)
    isempty(m) && throw(ArgumentError("no model: pass `model=` or construct FastFlowLM(\"tag\")"))
    j = chat_completion(
        b.client, m, to_messages(messages);
        temperature = temperature, max_tokens = max_tokens, stream = stream, on_token = on_token, kw...,
    )
    return parse_chat_response(j, m)
end
chat(b::FastFlowLM, messages::AbstractString; kw...) = chat(b, to_messages(messages); kw...)

"""Single-prompt completion (wraps the prompt as one user turn)."""
function complete(b::FastFlowLM, prompt::AbstractString; kw...)
    return chat(b, to_messages(prompt); kw...)
end

"""
    embed(b::FastFlowLM, input; model="")

Embedding vector(s) for `input` (a `String` or vector of `String`). Requires the server
to have been started with embedding support (`serve!(b; embed=true)` or `flm serve … --embed 1`).
"""
function embed(b::FastFlowLM, input; model::AbstractString = "")
    m = _model_or(b, model)
    j = embeddings(b.client, m, input)
    data = _dig(j, "data"; default = [])
    return [collect(Float64.(_dig(d, "embedding"; default = Float64[]))) for d in data]
end

# ── Model management (via the flm CLI) ───────────────────────────────────────

"""List models known to FastFlowLM (parsed from `flm list --json`)."""
function list_models(b::FastFlowLM)
    ok, out = _capture(`$(b.binary) list --json`)
    (ok && !isempty(out)) || return ModelInfo[]
    j = try
        JSON.parse(out)
    catch
        return ModelInfo[]
    end
    infos = ModelInfo[]
    for m in get(j, "models", [])
        push!(
            infos, ModelInfo(
                String(get(m, "name", get(m, "model", "?"))),
                Bool(get(m, "installed", false)),
                String(_dig(m, "details", "family"; default = "")),
                String(_dig(m, "details", "parameter_size"; default = "")),
                String(_dig(m, "details", "quantization_level"; default = "")),
                Int(get(m, "default_context_length", 0)),
                Float64(get(m, "footprint", 0.0)),
            ),
        )
    end
    return infos
end

"""Download a model by FastFlowLM tag (e.g. `"llama3.2:1b"`) via `flm pull`."""
function pull_model(b::FastFlowLM, tag::AbstractString; force::Bool = false)
    args = force ? `$(b.binary) pull $tag --force` : `$(b.binary) pull $tag`
    return success(pipeline(ignorestatus(args); stdout = stdout, stderr = stderr))
end

"""Backend + hardware readiness — delegates to [`tonalli_doctor`](@ref)."""
health(b::FastFlowLM) = tonalli_doctor(; show = false)

# ── Server lifecycle ─────────────────────────────────────────────────────────

"""
    serve!(b::FastFlowLM; model="", embed=false, wait=true, timeout=120)

Launch `flm serve` as a managed background process bound to `b.host`/`b.port`. When
`wait=true`, blocks until the server answers `/models` (up to `timeout` seconds). Returns `b`.
"""
function serve!(b::FastFlowLM; model::AbstractString = "", embed::Bool = false, wait::Bool = true, timeout::Integer = 120)
    m = _model_or(b, model)
    isempty(m) && throw(ArgumentError("no model to serve: pass `model=` or construct FastFlowLM(\"tag\")"))
    if b.proc[] !== nothing && process_running(b.proc[])
        return b
    end
    cmd = `$(b.binary) serve $m --host $(b.host) --port $(b.port) --pmode $(b.pmode) --quiet`
    embed && (cmd = `$cmd --embed 1`)
    b.proc[] = run(pipeline(ignorestatus(cmd); stdout = devnull, stderr = devnull); wait = false)
    if wait
        t0 = time()
        while time() - t0 < timeout
            ping(b.client) && return b
            process_running(b.proc[]) || error("flm serve exited before becoming ready")
            sleep(1.0)
        end
        error("flm serve did not become ready within $(timeout)s")
    end
    return b
end

"""Stop the managed `flm serve` process, if any."""
function stop!(b::FastFlowLM)
    p = b.proc[]
    if p !== nothing && process_running(p)
        kill(p)
    end
    b.proc[] = nothing
    return b
end
