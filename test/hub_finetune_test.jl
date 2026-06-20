@testitem "HFModel construction + resolve guard" begin
    using Tonalli
    using Tonalli: resolve
    m = HFModel("org/repo"; filename = "model.gguf", revision = "main")
    @test m.repo_id == "org/repo"
    @test m.filename == "model.gguf"
    # No filename → resolve must error rather than guess.
    @test_throws ErrorException resolve(HFModel("org/repo"))
end

@testitem "LoRAConfig kwargs + TOML" begin
    using Tonalli
    cfg = LoRAConfig(base_model = "m", dataset = "d.jsonl", rank = 8, epochs = 2)
    @test cfg.base_model == "m"
    @test cfg.rank == 8
    @test cfg.epochs == 2
    @test cfg.target_device == "rocm"

    mktemp() do path, io
        write(io, "[lora]\nbase_model = \"meta/x\"\ndataset = \"train.jsonl\"\nrank = 16\nepochs = 3\n")
        close(io)
        c2 = LoRAConfig(path)
        @test c2.base_model == "meta/x"
        @test c2.rank == 16
        @test c2.epochs == 3
    end
end

@testitem "CommandLineTuner runs an external CLI trainer" begin
    using Tonalli
    dir = mktempdir()
    cfg = LoRAConfig(base_model = "x", dataset = "y.jsonl", output_dir = dir)
    # The "trainer" is just a CLI tool (mkdir) creating the adapter dir — no Python.
    t = CommandLineTuner(cfg; command = c -> `mkdir -p $(joinpath(c.output_dir, "adapter"))`)
    out = finetune(t)
    @test out == abspath(joinpath(dir, "adapter"))
    @test isdir(out)
end

@testitem "CommandLineTuner surfaces a failing trainer" begin
    using Tonalli
    t = CommandLineTuner(LoRAConfig(base_model = "x", dataset = "y.jsonl"); command = c -> `false`)
    @test_throws ErrorException finetune(t)
end

@testitem "gguf metadata via real download" tags = [:network] begin
    using Tonalli
    # A tiny real GGUF on the Hub; only runs when TONALLI_TEST_NETWORK=1.
    path = hf_download("ggml-org/models", "tinyllamas/stories15M-q4_0.gguf")
    md = gguf_metadata(path)
    @test md isa Dict
    @test haskey(md, "general.architecture")
end
