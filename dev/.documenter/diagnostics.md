
# Diagnostics {#Diagnostics}

[`tonalli_doctor`](/api#Tonalli.tonalli_doctor) probes the entire stack and returns a [`HealthReport`](/api#Tonalli.HealthReport). It is read-only — it shells out to `flm validate --json`, reads `/proc` and `/dev`, and checks ROCm — so it is always safe to run.

```julia
using Tonalli
r = tonalli_doctor()          # prints a table; returns the report
r.ready                       # Bool: NPU inference go/no-go
r.checks                      # Vector{CheckResult}
```


## What it checks {#What-it-checks}

|                   Check |                        Meaning |                                       If it fails |
| -----------------------:| ------------------------------:| -------------------------------------------------:|
|            `flm binary` |           FastFlowLM on `PATH` |                       Install from fastflowlm.com |
|        `amdxdna driver` |           kernel module loaded |         Kernel 7.0+ with amdxdna, or amdxdna-dkms |
|            `NPU device` |    `/dev/accel/accel*` present |          Enable NPU in BIOS; check driver binding |
|          `flm validate` | firmware / kernel / memlock OK | Update firmware (≥1.1.0.0); `ulimit -l unlimited` |
| `ROCm iGPU (fine-tune)` |            gfx target for ROCm |       Optional; install ROCm to fine-tune locally |


`ready` requires the binary, the device, and FastFlowLM's own validation. The ROCm check is **advisory** — needed only for local fine-tuning, not inference.

A backend's [`health`](/api#Tonalli.health) delegates to the doctor (FastFlowLM) or pings the server (Lemonade/Ollama):

```julia
health(FastFlowLM())
health(OllamaBackend(; port = 11434))
```


## Programmatic use {#Programmatic-use}

```julia
r = tonalli_doctor(; show = false)
for c in r.checks
    c.ok || @warn c.name c.detail c.advice
end
```

