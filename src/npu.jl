# NPU / iGPU stack diagnostics. Everything here is read-only probing — shelling out to
# `flm validate --json`, reading /proc and /dev, and (advisory) ROCm detection — so it is
# safe to run anywhere and degrades gracefully when tools are missing.

using JSON
using Printf: @printf

"""Run `cmd`, capturing stdout; returns `(success::Bool, output::String)`."""
function _capture(cmd::Cmd)
    out = IOBuffer()
    try
        ok = success(pipeline(ignorestatus(cmd); stdout = out, stderr = devnull))
        return (ok, String(take!(out)))
    catch
        return (false, "")
    end
end

"""Locate the `flm` (FastFlowLM) binary, or `nothing`."""
flm_binary() = Sys.which("flm")

"""True if the amdxdna kernel module is loaded."""
function _amdxdna_loaded()
    isfile("/proc/modules") || return false
    return any(line -> startswith(line, "amdxdna "), eachline("/proc/modules"))
end

_first_accel_device() = begin
    dir = "/dev/accel"
    isdir(dir) || return nothing
    devs = filter(startswith("accel"), readdir(dir))
    isempty(devs) ? nothing : joinpath(dir, first(devs))
end

"""
    tonalli_doctor(; show = true) -> HealthReport

Probe the AMD NPU + iGPU stack and return a [`HealthReport`](@ref). Checks the `flm`
binary, the `amdxdna` driver and `/dev/accel` device, FastFlowLM's own NPU validation
(`flm validate`), and — advisory, for fine-tuning — ROCm/iGPU availability.

`HealthReport.ready` reflects inference readiness on the NPU. Pass `show = false` to
suppress printing.
"""
function tonalli_doctor(; show::Bool = true)
    checks = CheckResult[]

    # 1. flm binary
    flm = flm_binary()
    push!(
        checks, CheckResult(
            "flm binary",
            flm !== nothing,
            flm === nothing ? "not found on PATH" : flm,
            "Install FastFlowLM from https://fastflowlm.com/docs/install_lin/",
        ),
    )

    # 2. amdxdna driver
    drv = _amdxdna_loaded()
    push!(
        checks, CheckResult(
            "amdxdna driver",
            drv,
            drv ? "loaded" : "kernel module not loaded",
            "Need kernel 7.0+ with amdxdna, or install amdxdna-dkms.",
        ),
    )

    # 3. /dev/accel device
    dev = _first_accel_device()
    push!(
        checks, CheckResult(
            "NPU device",
            dev !== nothing,
            dev === nothing ? "no /dev/accel/accel* device" : dev,
            "Confirm the NPU is enabled in BIOS and amdxdna bound the device.",
        ),
    )

    # 4. FastFlowLM NPU validation (authoritative: firmware, kernel, memlock)
    validate_ready = false
    if flm !== nothing
        ok, out = _capture(`$flm validate --json`)
        if ok && !isempty(out)
            try
                j = JSON.parse(out)
                validate_ready = get(j, "ready", false)
                fw = get(j, "all_fw_ok", false)
                kok = get(j, "kernel_ok", false)
                mok = get(j, "memlock_ok", false)
                detail = "ready=$(validate_ready) fw_ok=$fw kernel_ok=$kok memlock_ok=$mok kernel=$(get(j, "kernel", "?"))"
                push!(
                    checks, CheckResult(
                        "flm validate", validate_ready, detail,
                        validate_ready ? "" :
                            "Update NPU firmware (>=1.1.0.0), kernel, or raise memlock (ulimit -l unlimited).",
                    ),
                )
            catch e
                push!(checks, CheckResult("flm validate", false, "unparseable output: $e", "Run `flm validate` manually."))
            end
        else
            push!(checks, CheckResult("flm validate", false, "command failed", "Run `flm validate` manually to see the error."))
        end
    else
        push!(checks, CheckResult("flm validate", false, "skipped (no flm)", "Install FastFlowLM first."))
    end

    # 5. ROCm / iGPU (advisory — needed only for local fine-tuning)
    rocminfo = Sys.which("rocminfo")
    if rocminfo !== nothing
        ok, out = _capture(`$rocminfo`)
        m = ok ? match(r"gfx\d+\w*", out) : nothing
        gfx = m === nothing ? "?" : m.match
        push!(
            checks, CheckResult(
                "ROCm iGPU (fine-tune)", ok && m !== nothing,
                ok ? "target $gfx" : "rocminfo failed",
                "Optional: required only for local LoRA fine-tuning. gfx115x iGPUs may need HSA_OVERRIDE_GFX_VERSION.",
            ),
        )
    else
        push!(
            checks, CheckResult(
                "ROCm iGPU (fine-tune)", false, "rocminfo not found",
                "Optional: install ROCm to fine-tune locally on the iGPU.",
            ),
        )
    end

    # Inference readiness = the binary, the device, and flm's own validation.
    ready = flm !== nothing && dev !== nothing && validate_ready
    report = HealthReport(ready, checks)
    show && print_report(report)
    return report
end

"""
    print_report(r::HealthReport; io = stdout)

Pretty-print a [`HealthReport`](@ref) with pass/fail markers and remediation advice.
"""
function print_report(r::HealthReport; io::IO = stdout)
    println(io, "Tonalli doctor — NPU/iGPU stack")
    println(io, "─"^60)
    for c in r.checks
        mark = c.ok ? "✓" : "✗"
        @printf(io, " %s  %-22s %s\n", mark, c.name, c.detail)
        if !c.ok && !isempty(c.advice)
            println(io, "      ↳ ", c.advice)
        end
    end
    println(io, "─"^60)
    println(io, r.ready ? "READY: NPU inference looks good." : "NOT READY: resolve the ✗ items above.")
    return r
end
