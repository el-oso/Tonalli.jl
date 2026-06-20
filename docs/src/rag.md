# Julia expert (RAG)

Tonalli ships a lightweight Retrieval-Augmented Generation (RAG) layer that turns a
collection of `.md` and `.jl` files into a searchable, citable knowledge base and
feeds the most relevant excerpts to any inference backend as grounded context.

## Quick start

```julia
using Tonalli

# 1. Build (or load) a vector store from your codebase / docs.
store = VectorStore()
b = FastFlowLM("gemma4-it:e4b")
serve!(b; embed = true)                            # one server, chat + embed

ingest!(store, b, ["src/", "docs/src/"]; model = "embed-gemma:300m")
save(store, "mystore.bin")

# 2. Ask a question — grounded in the ingested docs.
resp = ask(b, store, b, "How does multiple dispatch work in Julia?";
           k = 6, model = "gemma4-it:e4b")
println(resp.content)
println("Sources: ", resp.raw["rag_sources"])

stop!(b)
```

## Architecture

```
┌──────────────┐   ingest!   ┌─────────────────────────────────────┐
│ .md / .jl    │────────────▶│  VectorStore (dim × N Float32)      │
│  files       │  chunk +    │  chunks: Vector{Chunk}               │
└──────────────┘  embed +    │  embeddings: L2-normalised columns   │
                  normalise  └──────────────┬──────────────────────┘
                                            │ search (cosine via matmul)
                                            ▼
 query ──embed──▶ qvec ──────────────▶ top-k Chunks
                                            │
                              build_rag_prompt
                                            │
                                     generator.chat
                                            │
                                      ChatResponse
                                   (.raw["rag_sources"])
```

## API

### Data types

```@docs
Chunk
VectorStore
```

### Store operations

```@docs
add!
search
save
load_store
```

### Ingestion

```@docs
chunk_text
ingest!
```

### Retrieval and generation

```@docs
retrieve
build_rag_prompt
ask
```

## CLI

```
# Build or update a store from one or more paths (walks .md/.jl recursively):
tonalli ingest mystore.bin src/ docs/src/

# Ask a question (starts the server automatically if needed):
tonalli ask mystore.bin "What types does Julia support?"
```

The CLI uses `gemma4-it:e4b` for chat and `embed-gemma:300m` for embeddings. Both
are served from a single `flm serve --embed 1` process.

## Chunking

`chunk_text` splits on blank-line/paragraph and code-fence boundaries with configurable
`max_chars` (default 1200) and `overlap` (default 200 chars). For `.jl` files,
`ingest!` first extracts docstring + adjacent signature blocks to make chunks
semantically dense, then falls back to `chunk_text` for plain source files.

## Persistence

Stores are serialised with Julia's `Serialization` stdlib:

```julia
save(store, "mystore.bin")
store2 = load_store("mystore.bin")
```

The binary format is Julia-version-dependent; regenerate the store if you upgrade Julia.

## Grounding and citation

`build_rag_prompt` instructs the model to:

1. Answer **only** from the provided context excerpts.
2. Cite every fact with its `[source]` tag (e.g. `[src/foo.jl#3]`).
3. Admit when the context is insufficient rather than hallucinating.

The sources are also returned in `ChatResponse.raw["rag_sources"]` for programmatic use.
