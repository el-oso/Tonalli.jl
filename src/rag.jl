# Retrieval-Augmented Generation (RAG) — Julia-expert knowledge base.
#
# Design:
#   • `Chunk` is an immutable value with id, text, source, and open metadata.
#   • `VectorStore` holds L2-normalised Float32 column-vectors (dim × N) so that
#     cosine similarity reduces to a dot product — a single `embeddings' * q` call.
#   • `ingest!` splits .md/.jl files into chunks, embeds them in batches via the
#     `AbstractInferenceBackend` embed() interface, normalises, and appends.
#   • `ask` = retrieve → build_rag_prompt → chat; the call-site controls which
#     backend generates answers and which embeds queries.
#
# Persistence via the `Serialization` stdlib so nothing new goes in Project.toml.
# `Serialization` is a stdlib — add it to [deps] only if not already present.

using Serialization: serialize, deserialize
using LinearAlgebra: norm, normalize

# ── Value types ──────────────────────────────────────────────────────────────

"""
    Chunk(id, text, source, metadata)

An atomic retrievable text fragment. `source` is a human-readable provenance string
(e.g. the relative file path + chunk index). `metadata` is an open `Dict{String,Any}`
for any caller-supplied fields.
"""
struct Chunk
    id::String
    text::String
    source::String
    metadata::Dict{String, Any}
end

"""
    VectorStore

In-memory vector store backed by a `Float32` matrix (dim × N, L2-normalised columns).
Cosine similarity = embeddings' * normalise(query). Mutable so chunks and embeddings
can be appended without constructing a new object.
"""
mutable struct VectorStore
    chunks::Vector{Chunk}
    embeddings::Matrix{Float32}          # dim × N; each column is L2-normalised
    dim::Int
end

"""Construct an empty `VectorStore` with unknown dimension (dim = 0)."""
VectorStore() = VectorStore(Chunk[], Matrix{Float32}(undef, 0, 0), 0)

# ── VectorStore operations ────────────────────────────────────────────────────

"""
    add!(store, chunks, embs)

Append `chunks` (a `Vector{Chunk}`) and their embedding matrix `embs`
(dim × length(chunks)) to the store. Columns are L2-normalised before insertion.
`embs` may be `Matrix{Float32}` or `Matrix{Float64}` (converted automatically).
"""
function add!(store::VectorStore, chunks::Any, embs::Any)
    chunks_vec = chunks isa Vector{Chunk} ? chunks : Vector{Chunk}(chunks)
    isempty(chunks_vec) && return store
    E = Float32.(embs)                               # dim × n
    # L2-normalise each column in-place
    for j in axes(E, 2)
        col = view(E, :, j)
        n = norm(col)
        n > 0 && (col ./= n)
    end
    if store.dim == 0
        store.dim = size(E, 1)
        store.embeddings = E
    else
        size(E, 1) == store.dim || error(
            "embedding dim mismatch: store has $(store.dim), got $(size(E, 1))"
        )
        store.embeddings = hcat(store.embeddings, E)
    end
    append!(store.chunks, chunks_vec)
    return store
end

"""
    search(store, qvec, k) -> Vector{Tuple{Chunk, Float32}}

Return the top-`k` chunks by cosine similarity to `qvec` (a Float64 or Float32 vector),
sorted descending. Returns fewer than `k` results if the store has fewer than `k` chunks.
"""
function search(store::VectorStore, qvec, k::Int)::Vector{Tuple{Chunk, Float32}}
    isempty(store.chunks) && return Tuple{Chunk, Float32}[]
    q = Float32.(qvec)
    n = norm(q)
    n > 0 && (q ./= n)
    scores = store.embeddings' * q               # N-vector of dot products = cosine sims
    actual_k = min(k, length(store.chunks))
    top_idx = partialsortperm(scores, 1:actual_k; rev = true)
    return [(store.chunks[i], scores[i]) for i in top_idx]
end

"""Serialise the store to `path` using Julia's Serialization stdlib."""
function save(store::VectorStore, path::AbstractString)
    open(path, "w") do io
        serialize(io, store)
    end
    return path
end

"""
    load_store(path) -> VectorStore

Deserialise a `VectorStore` from `path`.
"""
function load_store(path::AbstractString)::VectorStore
    return open(path, "r") do io
        deserialize(io)
    end
end

# ── Text chunking ─────────────────────────────────────────────────────────────

"""
    chunk_text(s; max_chars=1200, overlap=200) -> Vector{String}

Split `s` into overlapping chunks of at most `max_chars` characters. Split boundaries
prefer blank-line (paragraph) separators, then code-fence lines (` ``` `), then forced
mid-paragraph splits. Each chunk after the first starts with the last `overlap` characters
of the previous chunk so that sentence-spanning context is preserved.
"""
function chunk_text(s::AbstractString; max_chars::Int = 1200, overlap::Int = 200)::Vector{String}
    isempty(s) && return String[]
    # Tokenise into "logical blocks": paragraphs and code fences.
    blocks = _split_blocks(s)
    chunks = String[]
    current = IOBuffer()
    current_len = 0
    last_overlap = ""

    flush_chunk!() = begin
        text = String(take!(current))
        if !isempty(strip(text))
            push!(chunks, text)
            # prepare overlap for the next chunk
            last_overlap = length(text) > overlap ? text[(end - overlap + 1):end] : text
        end
        current = IOBuffer()
        current_len = 0
        if !isempty(last_overlap)
            write(current, last_overlap)
            current_len = length(last_overlap)
        end
    end

    for block in blocks
        blen = length(block)
        if blen == 0
            continue
        end
        if current_len + blen > max_chars && current_len > 0
            flush_chunk!()
        end
        # If a single block exceeds max_chars, split it hard.
        if blen > max_chars
            pos = 1
            while pos <= blen
                segment = block[pos:min(pos + max_chars - 1, blen)]
                write(current, segment)
                current_len += length(segment)
                if current_len >= max_chars
                    flush_chunk!()
                end
                pos += length(segment)
            end
        else
            write(current, block)
            current_len += blen
        end
    end
    # flush whatever is left
    text = String(take!(current))
    !isempty(strip(text)) && push!(chunks, text)
    return chunks
end

# Split text into paragraph/code-fence logical blocks (each block keeps its trailing \n).
function _split_blocks(s::AbstractString)
    blocks = String[]
    buf = IOBuffer()
    in_fence = false
    for line in eachline(IOBuffer(s); keep = true)
        if startswith(strip(line), "```")
            if !in_fence
                # flush accumulated text before the fence
                txt = String(take!(buf))
                !isempty(txt) && push!(blocks, txt)
            end
            in_fence = !in_fence
            write(buf, line)
            if !in_fence
                # flush the fence block
                txt = String(take!(buf))
                !isempty(txt) && push!(blocks, txt)
            end
        elseif !in_fence && strip(line) == ""
            # blank line → paragraph break
            write(buf, line)
            txt = String(take!(buf))
            !isempty(txt) && push!(blocks, txt)
        else
            write(buf, line)
        end
    end
    txt = String(take!(buf))
    !isempty(txt) && push!(blocks, txt)
    return blocks
end

# ── Julia-file heuristics ─────────────────────────────────────────────────────

# Extract docstrings + adjacent function/method signatures from Julia source. Falls back to
# chunk_text for files that have no docstrings (e.g. pure data files).
function _chunk_julia(src::AbstractString; max_chars::Int = 1200, overlap::Int = 200)
    pieces = _extract_julia_pieces(src)
    if isempty(pieces)
        return chunk_text(src; max_chars = max_chars, overlap = overlap)
    end
    # chunk_text each piece independently (each is small enough in practice)
    out = String[]
    for p in pieces
        append!(out, chunk_text(p; max_chars = max_chars, overlap = overlap))
    end
    return out
end

# Very lightweight heuristic: collect contiguous docstring+definition blocks.
function _extract_julia_pieces(src::AbstractString)
    pieces = String[]
    buf = IOBuffer()
    in_doc = false
    blank_gap = 0

    for line in eachline(IOBuffer(src); keep = true)
        stripped = strip(line)
        is_docstring = startswith(stripped, "\"\"\"")
        is_def = any(
            startswith(stripped, kw) for
                kw in ("function ", "macro ", "struct ", "abstract type ", "const ", "# ")
        )

        if is_docstring || (in_doc && blank_gap <= 1)
            in_doc = true
            blank_gap = 0
            write(buf, line)
        elseif in_doc && is_def
            write(buf, line)
            blank_gap = 0
        elseif in_doc && stripped == ""
            blank_gap += 1
            if blank_gap > 2
                txt = String(take!(buf))
                !isempty(strip(txt)) && push!(pieces, txt)
                in_doc = false
                blank_gap = 0
            else
                write(buf, line)
            end
        else
            # Non-doc content: flush in-progress piece, continue accumulating plain code
            if in_doc
                txt = String(take!(buf))
                !isempty(strip(txt)) && push!(pieces, txt)
                in_doc = false
                blank_gap = 0
            end
            write(buf, line)
        end
    end
    txt = String(take!(buf))
    !isempty(strip(txt)) && push!(pieces, txt)
    return pieces
end

# ── Ingestion pipeline ────────────────────────────────────────────────────────

"""
    ingest!(store, embedder, paths; model="", batch=16)

Walk `paths` (a single path or a vector of paths) recursively for `.md` and `.jl` files,
chunk them, embed each batch via `embed(embedder, batch; model=model)`, and append to
`store`. `.jl` files are chunked by docstring+signature; `.md` files by paragraph/fence.

`source` fields use paths relative to the first element of `paths` (a single path is used
as-is; for a vector the common prefix is stripped). The chunk index is appended.
Returns `store`.
"""
function ingest!(
        store::VectorStore,
        embedder::AbstractInferenceBackend,
        paths;
        model::AbstractString = "",
        batch::Int = 16,
    )
    paths_vec = paths isa AbstractString ? [paths] : collect(paths)
    base = length(paths_vec) == 1 ? dirname(paths_vec[1]) : _common_prefix(paths_vec)

    for root in paths_vec
        _ingest_path!(store, embedder, root, base, model, batch)
    end
    return store
end

function _ingest_path!(store, embedder, path, base, model, batch)
    return if isdir(path)
        for entry in readdir(path; join = true, sort = true)
            _ingest_path!(store, embedder, entry, base, model, batch)
        end
    elseif isfile(path) && (endswith(path, ".md") || endswith(path, ".jl"))
        rel = _relpath(path, base)
        src = read(path, String)
        texts = endswith(path, ".jl") ? _chunk_julia(src) : chunk_text(src)
        isempty(texts) && return

        # Process in batches to limit memory.
        for batch_start in 1:batch:length(texts)
            batch_end = min(batch_start + batch - 1, length(texts))
            batch_texts = texts[batch_start:batch_end]
            vecs = embed(embedder, batch_texts; model = model)
            # vecs is Vector{Vector{Float64}} per the FastFlowLM contract
            isempty(vecs) && continue
            dim = length(vecs[1])
            E = Matrix{Float32}(undef, dim, length(vecs))
            for (j, v) in enumerate(vecs)
                E[:, j] = Float32.(v)
            end
            chunks = [
                Chunk(
                        _chunk_id(rel, batch_start + j - 2),
                        batch_texts[j],
                        "$rel#$(batch_start + j - 2)",
                        Dict{String, Any}(),
                    ) for j in 1:length(batch_texts)
            ]
            add!(store, chunks, E)
        end
    end
end

_chunk_id(rel, idx) = replace(rel, "/" => "_") * "_chunk$(idx)"

function _relpath(path::AbstractString, base::AbstractString)
    isempty(base) && return path
    # Ensure base ends with separator for clean stripping.
    bsep = endswith(base, "/") ? base : base * "/"
    startswith(path, bsep) && return path[(length(bsep) + 1):end]
    return path
end

function _common_prefix(paths::Vector)
    isempty(paths) && return ""
    ref = paths[1]
    for p in paths[2:end]
        while !isempty(ref) && !startswith(p, ref)
            ref = dirname(ref)
        end
    end
    return ref
end

# ── Retrieval ─────────────────────────────────────────────────────────────────

"""
    retrieve(store, embedder, query, k; model="") -> Vector{Chunk}

Embed `query` via `embedder` and return the top-`k` most similar `Chunk`s from `store`.
"""
function retrieve(
        store::VectorStore,
        embedder::AbstractInferenceBackend,
        query::AbstractString,
        k::Int;
        model::AbstractString = "",
    )::Vector{Chunk}
    vecs = embed(embedder, [query]; model = model)
    isempty(vecs) && return Chunk[]
    qvec = vecs[1]
    return [c for (c, _) in search(store, qvec, k)]
end

# ── Prompt construction ───────────────────────────────────────────────────────

# Sane budget: ~6 × 1200-char chunks ≈ 7200 chars of context + system preamble.
const _MAX_CONTEXT_CHARS = 8000

"""
    build_rag_prompt(query, chunks) -> Vector{ChatMessage}

Build a two-message prompt: a system message that instructs the model to answer as a Julia
expert grounded in the provided context (citing `source` tags, admitting gaps), and a user
message with the retrieved context blocks + the original question.

Total context is capped at `$(_MAX_CONTEXT_CHARS)` characters to stay within typical
context windows.
"""
function build_rag_prompt(query::AbstractString, chunks::AbstractVector)::Vector{ChatMessage}
    system_text = """
    You are a helpful Julia programming expert. Answer the user's question using ONLY the \
    context excerpts provided below. For every fact you state, cite its source in square \
    brackets, e.g. [src/foo.jl#3]. If the context does not contain sufficient information \
    to answer the question, say so explicitly — do not invent details.
    """

    # Build context block, respecting the character budget.
    ctx = IOBuffer()
    used = 0
    for (i, chunk) in enumerate(chunks)
        block = "[$(chunk.source)]\n$(chunk.text)\n\n"
        blen = length(block)
        if used + blen > _MAX_CONTEXT_CHARS
            break
        end
        write(ctx, block)
        used += blen
    end
    context_str = String(take!(ctx))

    user_text = if isempty(context_str)
        "Context: (none)\n\nQuestion: $query"
    else
        "Context:\n\n$(context_str)Question: $query"
    end

    return [ChatMessage("system", strip(system_text)), ChatMessage("user", user_text)]
end

# ── High-level ask ────────────────────────────────────────────────────────────

"""
    ask(generator, store, embedder, query; k=6, model="", kw...) -> ChatResponse

Full RAG pipeline: embed `query` with `embedder`, retrieve the top-`k` chunks from
`store`, build the grounded prompt, and generate an answer with `generator`.

`kw` is forwarded to `chat`. Use `model=` to override the model name on either backend.

The retrieved `Chunk` sources are attached to the returned `ChatResponse`'s `raw` field
under the key `"rag_sources"` (a `Vector{String}`).
"""
function ask(
        generator::AbstractInferenceBackend,
        store::VectorStore,
        embedder::AbstractInferenceBackend,
        query::AbstractString;
        k::Int = 6,
        model::AbstractString = "",
        kw...,
    )::ChatResponse
    chunks = retrieve(store, embedder, query, k; model = model)
    messages = build_rag_prompt(query, chunks)
    sources = [c.source for c in chunks]
    resp = chat(generator, messages; model = model, kw...)
    # Attach sources to the raw field — copy the raw dict and add the key.
    raw_with_sources = if resp.raw isa AbstractDict
        d = Dict{String, Any}(string(k) => v for (k, v) in resp.raw)
        d["rag_sources"] = sources
        d
    else
        Dict{String, Any}("rag_sources" => sources)
    end
    return ChatResponse(resp.content, resp.model, resp.finish_reason, resp.usage, raw_with_sources)
end
