# Tests for src/rag.jl — pure Julia, no NPU required for the unit tests.
# The `:npu` test at the bottom is auto-skipped when /dev/accel is absent.

# ── Mock backend ─────────────────────────────────────────────────────────────
# Defined at module scope so every @testitem can reference it without
# re-defining or reopening the struct.

# ── Unit tests ────────────────────────────────────────────────────────────────

@testitem "chunk_text: basic paragraph split" begin
    using Tonalli: chunk_text

    text = join(["A"^600, "B"^600, "C"^600], "\n\n")
    chunks = chunk_text(text; max_chars = 700, overlap = 50)
    @test length(chunks) >= 3
    # Each chunk fits within max_chars + a bit of overlap
    for c in chunks
        @test length(c) <= 1800   # generous upper bound
    end
    # overlap: the second chunk starts with the tail of the first
    @test length(chunks) >= 2 && occursin(first(chunks)[(end - 49):end], chunks[2])
end

@testitem "chunk_text: code-fence boundary" begin
    using Tonalli: chunk_text

    src = """
    Some prose before the fence.

    ```julia
    x = 1
    y = 2
    ```

    Some prose after the fence.
    """
    chunks = chunk_text(src; max_chars = 60, overlap = 10)
    @test !isempty(chunks)
    # At least one chunk should contain the fence marker
    @test any(c -> occursin("```", c), chunks)
end

@testitem "chunk_text: empty input" begin
    using Tonalli: chunk_text
    @test chunk_text("") == String[]
    @test chunk_text("   \n\n  ") == String[]
end

@testitem "VectorStore add! and search" begin
    using Tonalli: Chunk, VectorStore, add!, search

    store = VectorStore()
    dim = 4
    # Three chunks with known embeddings (will be L2-normalised inside add!)
    chunks = [
        Chunk("c1", "Julia is fast", "file1.md#1", Dict{String, Any}()),
        Chunk("c2", "Python is popular", "file1.md#2", Dict{String, Any}()),
        Chunk("c3", "Julia is open source", "file1.md#3", Dict{String, Any}()),
    ]
    embs = Float32[
        1  0  0.9;   # dim 1
        0  1  0  ;
        0  0  0.1;
        0  0  0  ;
    ]
    add!(store, chunks, embs)

    @test length(store.chunks) == 3
    @test store.dim == 4

    # Query pointing along dim-1 should prefer c1 and c3 over c2
    qvec = Float32[1.0, 0.0, 0.0, 0.0]
    results = search(store, qvec, 3)
    @test length(results) == 3
    @test results[1][1].id == "c1"       # highest cosine sim on dim-1
    @test results[end][1].id == "c2"     # c2 lives on dim-2
    # All scores should be in [-1, 1]
    for (_, score) in results
        @test -1 <= score <= 1
    end
end

@testitem "VectorStore add! dimensionality mismatch" begin
    using Tonalli: Chunk, VectorStore, add!
    store = VectorStore()
    c1 = [Chunk("a", "t", "s", Dict{String, Any}())]
    add!(store, c1, Float32[1.0; 0.0;;])
    c2 = [Chunk("b", "t", "s", Dict{String, Any}())]
    @test_throws ErrorException add!(store, c2, Float32[1.0; 0.0; 0.0;;])
end

@testitem "VectorStore search empty store" begin
    using Tonalli: VectorStore, search
    store = VectorStore()
    @test search(store, Float32[1.0, 0.0], 5) == []
end

@testitem "VectorStore save and load_store roundtrip" begin
    using Tonalli: Chunk, VectorStore, add!, save, load_store
    store = VectorStore()
    chunks = [Chunk("x", "hello world", "src/test.jl#1", Dict{String, Any}("foo" => 42))]
    add!(store, chunks, Float32[0.6; 0.8;;])
    dir = mktempdir()
    path = joinpath(dir, "store.bin")
    save(store, path)
    @test isfile(path)
    loaded = load_store(path)
    @test loaded.dim == 2
    @test length(loaded.chunks) == 1
    @test loaded.chunks[1].id == "x"
    @test loaded.chunks[1].metadata["foo"] == 42
    # embeddings should be bit-for-bit identical after roundtrip
    @test loaded.embeddings ≈ store.embeddings
end

@testitem "build_rag_prompt structure" begin
    using Tonalli: Chunk, build_rag_prompt, ChatMessage

    chunks = [
        Chunk("a", "Julia is dynamically typed.", "docs/types.md#1", Dict{String, Any}()),
        Chunk("b", "Julia uses multiple dispatch.", "docs/dispatch.md#1", Dict{String, Any}()),
    ]
    msgs = build_rag_prompt("What is Julia?", chunks)
    @test length(msgs) == 2
    @test msgs[1].role == "system"
    @test msgs[2].role == "user"
    sys = msgs[1].content
    usr = msgs[2].content
    # System message should mention citing sources
    @test occursin("source", sys) || occursin("cite", sys)
    # User message should contain the retrieved context
    @test occursin("Julia is dynamically typed", usr)
    @test occursin("docs/types.md#1", usr)
    @test occursin("What is Julia?", usr)
end

@testitem "build_rag_prompt with empty chunks" begin
    using Tonalli: Chunk, build_rag_prompt

    msgs = build_rag_prompt("What is Julia?", Chunk[])
    @test length(msgs) == 2
    @test occursin("What is Julia?", msgs[2].content)
end

@testitem "mock backend: ingest! + retrieve + ask" begin
    using Tonalli:
        AbstractInferenceBackend,
        ChatMessage, ChatResponse, Usage, ModelInfo, HealthReport, CheckResult,
        VectorStore, Chunk, add!, search, ingest!, retrieve, build_rag_prompt, ask

    # Minimal deterministic mock: embed maps each string to a fixed-dim hash-derived vector,
    # chat echoes the first context source from the user message.
    struct MockBackend <: AbstractInferenceBackend end

    function Tonalli.embed(::MockBackend, input; kw...)
        function _hash_vec(s)
            h = hash(s)
            Float64[
                sin(Float64(h)),
                cos(Float64(h)),
                sin(Float64(h >> 16)),
                cos(Float64(h >> 16)),
            ]
        end
        texts = input isa AbstractString ? [input] : collect(input)
        return [_hash_vec(t) for t in texts]
    end

    function Tonalli.chat(::MockBackend, msgs::AbstractVector; kw...)
        user_content = ""
        for m in msgs
            if m isa ChatMessage && m.role == "user"
                user_content = m.content
            end
        end
        # Extract first source tag from context if present
        m = match(r"\[([^\]]+)\]", user_content)
        answer = m === nothing ? "no context found" : "answered using $(m.captures[1])"
        return ChatResponse(answer, "mock", "stop", nothing, Dict{String, Any}())
    end

    # Required contract methods (trivial stubs)
    Tonalli.complete(b::MockBackend, p::AbstractString; kw...) =
        Tonalli.chat(b, [ChatMessage("user", p)]; kw...)
    Tonalli.list_models(::MockBackend) = ModelInfo[]
    Tonalli.pull_model(::MockBackend, ::AbstractString) = true
    Tonalli.health(::MockBackend) = HealthReport(true, CheckResult[])

    dir = mktempdir()
    # Write a couple of tiny markdown documents
    write(joinpath(dir, "a.md"), "# Alpha\n\nAlpha is the first Greek letter.\n")
    write(joinpath(dir, "b.md"), "# Beta\n\nBeta is the second Greek letter.\n")

    backend = MockBackend()
    store = VectorStore()
    ingest!(store, backend, dir)

    @test length(store.chunks) >= 2

    # retrieve should give back chunks (ordering is hash-based, just check non-empty)
    hits = retrieve(store, backend, "What is Alpha?", 2)
    @test !isempty(hits)
    @test hits[1] isa Chunk

    # build_rag_prompt should include context
    msgs = build_rag_prompt("test?", hits)
    @test occursin("Alpha", msgs[2].content) || occursin("Beta", msgs[2].content)

    # ask should return a ChatResponse with rag_sources populated
    resp = ask(backend, store, backend, "What is Alpha?"; k = 2)
    @test resp isa ChatResponse
    @test !isempty(resp.content)
    sources = get(resp.raw, "rag_sources", nothing)
    @test sources isa Vector
    @test !isempty(sources)
end

@testitem "contracts: VectorStore satisfies AbstractVectorStore" begin
    using Tonalli: VectorStore, AbstractVectorStore
    using TypeContracts: satisfies
    res = satisfies(VectorStore, AbstractVectorStore)
    @test res.satisfied || error("VectorStore missing: $(res.missing_methods)")
end

# ── Live NPU test (auto-skipped when no /dev/accel) ──────────────────────────

@testitem "RAG live NPU: ingest tiny docs + ask" tags = [:npu] begin
    using Tonalli

    # Pull the embed model (quiet, no TTY)
    run(ignorestatus(`flm pull embed-gemma:300m --quiet`))

    # Write 3 tiny Julia fact files to a tempdir
    dir = mktempdir()
    write(
        joinpath(dir, "fact1.md"), """
        # Julia multiple dispatch
        Julia selects the method implementation to call based on the types of all function arguments, not just the first one. This is called multiple dispatch.
        """
    )
    write(
        joinpath(dir, "fact2.md"), """
        # Julia type system
        Julia has a rich type hierarchy. Every value in Julia has a type. Types can be abstract or concrete. Concrete types can be instantiated; abstract types cannot.
        """
    )
    write(
        joinpath(dir, "fact3.md"), """
        # Julia performance
        Julia is designed for high performance. Julia programs compile to efficient native code through LLVM. Performance is comparable to C and Fortran for numerical workloads.
        """
    )

    # Serve gemma4-it:e4b with embed support (single server handles both roles)
    b = FastFlowLM("gemma4-it:e4b")
    started = false
    if !Tonalli.ping(b.client)
        serve!(b; embed = true)
        started = true
    end
    try
        store = VectorStore()
        ingest!(store, b, dir; model = "embed-gemma:300m")
        @test length(store.chunks) >= 3

        resp = ask(b, store, b, "How does Julia dispatch methods?"; k = 3, model = "gemma4-it:e4b")
        @test resp isa ChatResponse
        @test !isempty(resp.content)
        sources = get(resp.raw, "rag_sources", String[])
        @test !isempty(sources)
        println("RAG answer: ", first(resp.content, 200))
        println("Sources: ", sources)
    finally
        started && stop!(b)
    end
end
