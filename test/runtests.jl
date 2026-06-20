# Run with: julia --project=test test/runtests.jl
using Tonalli
using ReTestItems

# :npu tests need a real XDNA device; :network tests hit the internet. Skip them unless
# the environment supports them (mirrors Mexicah's :matlab tag gating).
const NPU_AVAILABLE = isdir("/dev/accel") && !isempty(filter(startswith("accel"), readdir("/dev/accel")))
const RUN_NETWORK = get(ENV, "TONALLI_TEST_NETWORK", "0") == "1"

runtests(
    ti -> !((!NPU_AVAILABLE && :npu in ti.tags) || (!RUN_NETWORK && :network in ti.tags)),
    Tonalli;
    testitem_timeout = 180,
    nworkers = 0,
)
