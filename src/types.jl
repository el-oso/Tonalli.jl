# Core value types shared across backends. Kept dependency-free and concrete so the
# interface contracts in `contracts.jl` can name them as return types.

"""
    ChatMessage(role, content)

A role-tagged chat turn. `role` is one of `"system"`, `"user"`, `"assistant"`, `"tool"`.

Convenience constructors accept a `Pair`, so `ChatMessage(:user => "hi")` works.
"""
struct ChatMessage
    role::String
    content::String
end
ChatMessage(p::Pair) = ChatMessage(String(first(p)), String(last(p)))

"""Token accounting returned by a backend, when available."""
struct Usage
    prompt_tokens::Int
    completion_tokens::Int
    total_tokens::Int
end

"""
    ChatResponse

Result of a single completion. `raw` holds the backend's parsed JSON for callers that
need fields Tonalli does not model.
"""
struct ChatResponse
    content::String
    model::String
    finish_reason::String
    usage::Union{Usage, Nothing}
    raw::Any
end

Base.show(io::IO, r::ChatResponse) =
    print(io, "ChatResponse(", repr(first(r.content, 60)), r.finish_reason == "" ? "" : ", $(r.finish_reason)", ")")

"""
    ModelInfo

Metadata about a model known to a backend (mirrors FastFlowLM's `list --json`).
"""
struct ModelInfo
    id::String
    installed::Bool
    family::String
    parameter_size::String
    quantization::String
    context_length::Int
    footprint_gb::Float64
end

"""One diagnostic probe: a named check with pass/fail, detail, and remediation advice."""
struct CheckResult
    name::String
    ok::Bool
    detail::String
    advice::String
end

"""
    HealthReport(ready, checks)

Aggregate diagnostic report. `ready` is the overall go/no-go for inference; `checks`
carries the individual [`CheckResult`](@ref)s. Render with [`print_report`](@ref).
"""
struct HealthReport
    ready::Bool
    checks::Vector{CheckResult}
end
