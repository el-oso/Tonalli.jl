# Fine-tuning extension — loaded when PythonCall is available. Drives a LoRA SFT run on
# the local AMD iGPU (ROCm) or CPU via PyTorch + transformers + peft + trl, then writes a
# safetensors adapter. This is the "train on iGPU/CPU, serve on NPU" path: the resulting
# adapter is merged into the base and converted for FastFlowLM serving (see docs).

module TonalliFineTuneExt

using Tonalli
using PythonCall

# Import the Python stack lazily with an actionable error if the env is incomplete.
function _pyimport_stack()
    mods = Dict{Symbol, Any}()
    for name in (:torch, :transformers, :peft, :trl, :datasets)
        try
            mods[name] = pyimport(String(name))
        catch e
            error(
                "Tonalli fine-tuning needs the Python package `$(name)`. Install a ROCm " *
                    "PyTorch env with: pip install transformers peft trl datasets, and a " *
                    "ROCm build of torch. Underlying error: $(e)"
            )
        end
    end
    return mods
end

function Tonalli._finetune_impl(t::Tonalli.ROCmLoRATuner)
    cfg = t.config
    m = _pyimport_stack()
    torch, transformers, peft, trl, datasets = m[:torch], m[:transformers], m[:peft], m[:trl], m[:datasets]

    device = cfg.target_device == "cpu" ? "cpu" : "cuda"  # ROCm torch presents as "cuda"
    if cfg.target_device == "rocm" && Bool(pytruth(torch.cuda.is_available())) == false
        @warn "ROCm/torch reports no GPU; falling back to CPU (slow). Check ROCm + HSA_OVERRIDE_GFX_VERSION."
        device = "cpu"
    end

    @info "Loading base model" cfg.base_model device
    tok = transformers.AutoTokenizer.from_pretrained(cfg.base_model)
    Bool(pytruth(tok.pad_token == pybuiltins.None)) && (tok.pad_token = tok.eos_token)

    dtype = device == "cpu" ? torch.float32 : torch.bfloat16
    model = transformers.AutoModelForCausalLM.from_pretrained(cfg.base_model; torch_dtype = dtype)
    model.to(device)

    lora = peft.LoraConfig(;
        r = cfg.rank, lora_alpha = cfg.alpha, lora_dropout = cfg.dropout,
        bias = "none", task_type = "CAUSAL_LM",
    )

    ds = datasets.load_dataset("json"; data_files = cfg.dataset, split = "train")

    sft_cfg = trl.SFTConfig(;
        output_dir = cfg.output_dir,
        num_train_epochs = cfg.epochs,
        learning_rate = cfg.learning_rate,
        max_seq_length = cfg.max_seq_len,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        logging_steps = 10,
        save_strategy = "epoch",
        bf16 = device != "cpu",
    )

    trainer = trl.SFTTrainer(;
        model = model, train_dataset = ds, peft_config = lora,
        processing_class = tok, args = sft_cfg,
    )
    @info "Starting LoRA fine-tune" epochs = cfg.epochs rank = cfg.rank
    trainer.train()

    adapter_dir = joinpath(cfg.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    @info "Adapter saved" adapter_dir
    return String(adapter_dir)
end

end # module TonalliFineTuneExt
