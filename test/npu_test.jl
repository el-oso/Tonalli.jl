@testitem "doctor returns a well-formed report" begin
    using Tonalli
    r = tonalli_doctor(; show = false)
    @test r isa HealthReport
    @test !isempty(r.checks)
    @test all(c -> c isa CheckResult, r.checks)
    @test any(c -> c.name == "flm binary", r.checks)
end

@testitem "print_report does not error" begin
    using Tonalli
    r = HealthReport(false, [CheckResult("x", false, "detail", "advice"), CheckResult("y", true, "ok", "")])
    io = IOBuffer()
    print_report(r; io = io)
    s = String(take!(io))
    @test occursin("NOT READY", s)
    @test occursin("advice", s)
end

@testitem "NPU stack is ready" tags = [:npu] begin
    using Tonalli
    r = tonalli_doctor(; show = false)
    @test r.ready
end
