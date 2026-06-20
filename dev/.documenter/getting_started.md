
# Getting started {#Getting-started}

## Hardware & system requirements {#Hardware-and-system-requirements}

Tonalli's NPU path targets **AMD Ryzen AI chips with an XDNA 2 NPU** (Ryzen AI 300-series / Strix / Krackan / Kraken). Ryzen AI 7000/8000/200-series (XDNA 1) are **not** supported by the FastFlowLM backend.

You need, on Linux:
- **Kernel 7.0+ with `amdxdna`**, or the `amdxdna-dkms` package, exposing `/dev/accel/accel0`.
  
- The **AMD XRT** runtime (`libxrt2`, `libxrt-npu2`).
  
- **NPU firmware ≥ 1.1.0.0** and `memlock` raised (`ulimit -l unlimited`).
  
- **[FastFlowLM](https://fastflowlm.com/docs/install_lin/)** (`flm` on `PATH`).
  
- For **local fine-tuning**: ROCm + a ROCm build of PyTorch on the iGPU (gfx115x).
  

## Install {#Install}

```julia
pkg> add https://github.com/el-oso/Tonalli.jl
```


## Verify your stack {#Verify-your-stack}

```julia
using Tonalli
tonalli_doctor()
```


```
Tonalli doctor — NPU/iGPU stack
────────────────────────────────────────────────────────────
 ✓  flm binary             /usr/bin/flm
 ✓  amdxdna driver         loaded
 ✓  NPU device             /dev/accel/accel0
 ✓  flm validate           ready=true fw_ok=true kernel_ok=true memlock_ok=true
 ✓  ROCm iGPU (fine-tune)  target gfx1152
────────────────────────────────────────────────────────────
READY: NPU inference looks good.
```


Each ✗ comes with remediation advice. See [Diagnostics](/diagnostics).

## First chat {#First-chat}

```julia
b = FastFlowLM("llama3.2:1b")
pull_model(b, "llama3.2:1b")     # one-time download
serve!(b)                        # launch the NPU server
r = chat(b, "Say hi in 3 words.")
println(r.content)
stop!(b)
```


## The CLI {#The-CLI}

Tonalli ships a `tonalli` app:

```bash
tonalli doctor
tonalli pull llama3.2:1b
tonalli chat llama3.2:1b "Explain XDNA in one sentence."
tonalli serve llama3.2:1b --port 8000
```


Install the launcher with `julia -e 'using Pkg; Pkg.Apps.develop(path="."))'`.
