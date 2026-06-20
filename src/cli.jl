# `tonalli` command-line app. Wired through Base.@main + [apps.tonalli] in Project.toml,
# mirroring Mexicah's CLI shape.

const _CLI_USAGE = """
tonalli — run & fine-tune LLMs on AMD Ryzen AI (NPU + iGPU)

Usage: tonalli <command> [args]

Commands:
  doctor                     Check the NPU/iGPU stack (driver, firmware, ROCm)
  list                       List FastFlowLM models (installed + available)
  pull <tag>                 Download a FastFlowLM model (e.g. llama3.2:1b)
  serve <tag> [--port N]     Start an NPU server for <tag> and keep it running
  chat <tag> [prompt...]     Chat with <tag> (one-shot if prompt given, else REPL)
  finetune <config.toml>     Run a LoRA fine-tune via an external CLI trainer
  help                       Show this message
"""

function (@main)(args::Vector{String})::Cint
    return _cli_run(args)
end

function _cli_run(args::Vector{String})::Cint
    isempty(args) && (println(_CLI_USAGE); return 0)
    cmd = args[1]
    rest = args[2:end]
    try
        cmd == "doctor" && return _cli_doctor()
        cmd == "list" && return _cli_list()
        cmd == "pull" && return _cli_pull(rest)
        cmd == "serve" && return _cli_serve(rest)
        cmd == "chat" && return _cli_chat(rest)
        cmd in ("finetune", "tune") && return _cli_finetune(rest)
        cmd in ("help", "-h", "--help") && (println(_CLI_USAGE); return 0)
    catch e
        println(stderr, "error: ", sprint(showerror, e))
        return 1
    end
    println(stderr, "unknown command: $cmd\n")
    println(_CLI_USAGE)
    return 2
end

function _cli_doctor()::Cint
    r = tonalli_doctor(; show = true)
    return r.ready ? 0 : 1
end

function _cli_list()::Cint
    for m in list_models(FastFlowLM())
        flag = m.installed ? "[installed]" : "[available]"
        println(rpad(m.id, 32), " ", flag, "  ", m.parameter_size, " ", m.quantization)
    end
    return 0
end

function _cli_pull(rest::Vector{String})::Cint
    isempty(rest) && (println(stderr, "usage: tonalli pull <tag>"); return 2)
    return pull_model(FastFlowLM(), rest[1]) ? 0 : 1
end

# Parse a trailing `--port N` out of args, returning (port_or_nothing, remaining).
function _take_port(rest::Vector{String})
    i = findfirst(==("--port"), rest)
    i === nothing && return (nothing, rest)
    port = parse(Int, rest[i + 1])
    return (port, vcat(rest[1:(i - 1)], rest[(i + 2):end]))
end

function _cli_serve(rest::Vector{String})::Cint
    port, rest = _take_port(rest)
    isempty(rest) && (println(stderr, "usage: tonalli serve <tag> [--port N]"); return 2)
    b = port === nothing ? FastFlowLM(rest[1]) : FastFlowLM(rest[1]; port = port)
    println("Starting NPU server for $(rest[1]) on $(b.host):$(b.port) …")
    serve!(b)
    println("Ready. Ctrl-C to stop.")
    try
        while true
            sleep(3600)
        end
    catch
        stop!(b)
    end
    return 0
end

function _cli_chat(rest::Vector{String})::Cint
    port, rest = _take_port(rest)
    isempty(rest) && (println(stderr, "usage: tonalli chat <tag> [prompt...]"); return 2)
    tag = rest[1]
    b = port === nothing ? FastFlowLM(tag) : FastFlowLM(tag; port = port)
    started = false
    if !ping(b.client)
        println(stderr, "(no server on $(b.host):$(b.port); starting one…)")
        serve!(b)
        started = true
    end
    try
        if length(rest) > 1
            prompt = join(rest[2:end], " ")
            chat(b, prompt; stream = true, on_token = t -> print(t))
            println()
        else
            println("Chatting with $tag — empty line or Ctrl-D to quit.")
            history = ChatMessage[]
            while true
                print("\n> ")
                line = readline()
                (isempty(line) || eof(stdin)) && break
                push!(history, ChatMessage("user", line))
                r = chat(b, copy(history); stream = true, on_token = t -> print(t))
                println()
                push!(history, ChatMessage("assistant", r.content))
            end
        end
    finally
        started && stop!(b)
    end
    return 0
end

function _cli_finetune(rest::Vector{String})::Cint
    isempty(rest) && (println(stderr, "usage: tonalli finetune <config.toml>"); return 2)
    cfg = LoRAConfig(rest[1])
    t = TOML.parsefile(rest[1])
    lora = get(t, "lora", t)
    trainer = get(lora, "trainer", nothing)
    if trainer === nothing
        println(
            stderr,
            """
            No `trainer` command in [lora]. Tonalli bundles no trainer (no Python by policy);
            point it at an external CLI trainer, e.g.:

              [lora]
              base_model = "..."
              dataset = "train.jsonl"
              trainer = ["my-trainer", "--model", "{base_model}", "--data", "{dataset}", "--out", "{output_dir}", "--rank", "{rank}"]

            Placeholders: {base_model} {dataset} {output_dir} {rank} {alpha} {epochs} {lr} {max_seq_len} {device}.
            Or drive a CommandLineTuner programmatically. A pure-Julia trainer is on the roadmap.
            """
        )
        return 1
    end
    out = finetune(CommandLineTuner(cfg; command = c -> _build_trainer_cmd(trainer, c)))
    println("Adapter written to: ", out)
    return 0
end

# Substitute config placeholders into a TOML-provided trainer argv (array of strings, or a
# whitespace-split string) and build a Cmd. No shell — direct argv, so paths are safe.
function _build_trainer_cmd(trainer, c::LoRAConfig)
    subs = [
        "{base_model}" => c.base_model, "{dataset}" => c.dataset, "{output_dir}" => c.output_dir,
        "{rank}" => string(c.rank), "{alpha}" => string(c.alpha), "{epochs}" => string(c.epochs),
        "{lr}" => string(c.learning_rate), "{max_seq_len}" => string(c.max_seq_len), "{device}" => c.target_device,
    ]
    argv = trainer isa AbstractString ? split(trainer) : trainer
    return Cmd(String[replace(string(a), subs...) for a in argv])
end
