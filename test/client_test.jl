@testitem "message normalization" begin
    using Tonalli: ChatMessage, to_messages
    @test to_messages("hi") == [ChatMessage("user", "hi")]
    @test to_messages(ChatMessage(:user => "hi")) == [ChatMessage("user", "hi")]
    msgs = to_messages([ChatMessage("system", "s"), :user => "u", Dict("role" => "assistant", "content" => "a")])
    @test length(msgs) == 3
    @test msgs[1].role == "system"
    @test msgs[2] == ChatMessage("user", "u")
    @test msgs[3] == ChatMessage("assistant", "a")
end

@testitem "client + FastFlowLM over a mock OpenAI server" begin
    using Tonalli
    using HTTP, JSON, Sockets

    function freeport()
        s = Sockets.listen(Sockets.localhost, 0)
        p = Int(Sockets.getsockname(s)[2])
        close(s)
        return p
    end

    # Streaming server (stream=true) so the SSE chat path is exercised the same way a
    # real FastFlowLM/Ollama server streams (chunked text/event-stream).
    function handler(http::HTTP.Stream)
        target = http.message.target
        reqbody = String(read(http))
        body = isempty(reqbody) ? Dict{String, Any}() : JSON.parse(reqbody)
        _send(s, ct = "application/json") = begin
            HTTP.setstatus(http, 200)
            HTTP.setheader(http, "Content-Type" => ct)
            HTTP.startwrite(http)
            write(http, s)
        end
        if occursin("/chat/completions", target)
            if get(body, "stream", false) === true
                HTTP.setstatus(http, 200)
                HTTP.setheader(http, "Content-Type" => "text/event-stream")
                HTTP.startwrite(http)
                write(http, string("data: ", JSON.json(Dict("choices" => [Dict("delta" => Dict("content" => "Hello"))])), "\n\n"))
                write(http, string("data: ", JSON.json(Dict("choices" => [Dict("delta" => Dict("content" => " world"), "finish_reason" => "stop")])), "\n\n"))
                write(http, "data: [DONE]\n\n")
                return
            end
            resp = Dict(
                "model" => "mock",
                "choices" => [Dict("message" => Dict("role" => "assistant", "content" => "Hi there"), "finish_reason" => "stop")],
                "usage" => Dict("prompt_tokens" => 3, "completion_tokens" => 2, "total_tokens" => 5),
            )
            _send(JSON.json(resp))
        elseif occursin("/embeddings", target)
            _send(JSON.json(Dict("data" => [Dict("embedding" => [0.1, 0.2, 0.3])])))
        elseif occursin("/models", target)
            _send(JSON.json(Dict("data" => [Dict("id" => "mock-model")])))
        else
            HTTP.setstatus(http, 404)
            HTTP.startwrite(http)
        end
        return
    end

    port = freeport()
    server = HTTP.serve!(handler, Sockets.localhost, port; stream = true)
    try
        b = FastFlowLM("mock-model"; host = "127.0.0.1", port = port, binary = "flm")

        r = chat(b, "hi")
        @test r.content == "Hi there"
        @test r.finish_reason == "stop"
        @test r.usage.total_tokens == 5

        r2 = complete(b, "hello")
        @test r2.content == "Hi there"

        e = embed(b, "hi")
        @test e[1] == [0.1, 0.2, 0.3]

        toks = String[]
        rs = chat(b, "hi"; stream = true, on_token = t -> push!(toks, t))
        @test join(toks) == "Hello world"
        @test rs.content == "Hello world"
        @test rs.finish_reason == "stop"
    finally
        close(server)
    end
end
