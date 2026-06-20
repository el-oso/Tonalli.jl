# API reference

## Interfaces

```@docs
AbstractInferenceBackend
AbstractModelSource
AbstractFineTuner
```

## Backends

```@docs
FastFlowLM
LemonadeBackend
OllamaBackend
```

## Inference verbs

```@docs
chat
complete
embed
list_models
pull_model
serve!
stop!
health
```

## Diagnostics

```@docs
tonalli_doctor
print_report
HealthReport
CheckResult
```

## Hub

```@docs
hf_download
gguf_metadata
HFModel
```

## Fine-tuning

```@docs
LoRAConfig
CommandLineTuner
finetune
```

## Value types

```@docs
ChatMessage
ChatResponse
ModelInfo
Usage
```
