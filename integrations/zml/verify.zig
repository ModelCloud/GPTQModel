// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const zml = @import("zml");
const window = @import("qvq_window.zig");

fn run(input: window.Input, config: window.Config) zml.Tensor {
    return window.linear(input, config);
}

const ErrorStats = struct {
    mean: f32,
    max: f32,
    exact: bool,
    accepted: bool,
};

fn f16At(bytes: []const u8, index: usize) f32 {
    const offset = index * 2;
    const bits: u16 = @as(u16, bytes[offset]) |
        (@as(u16, bytes[offset + 1]) << 8);
    return @floatCast(@as(f16, @bitCast(bits)));
}

fn compareF16(actual: []const u8, expected: []const u8, require_exact: bool) !ErrorStats {
    if (actual.len != expected.len or actual.len % 2 != 0 or actual.len == 0)
        return error.InvalidExpectedOutput;
    const count = actual.len / 2;
    var sum: f64 = 0;
    var maximum: f32 = 0;
    var exact = true;
    for (0..count) |index| {
        const difference = f16At(actual, index) - f16At(expected, index);
        const absolute: f32 = if (difference < 0) -difference else difference;
        sum += @floatCast(absolute);
        if (absolute > maximum) maximum = absolute;
        if (absolute != 0) exact = false;
    }
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(count)));
    const accepted = exact or (!require_exact and mean <= 2e-3 and maximum <= 3.0 / 64.0);
    return .{ .mean = mean, .max = maximum, .exact = exact, .accepted = accepted };
}

fn measureCandidate(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    tensors: window.Input,
    buffers: zml.Bufferized(window.Input),
    candidate: window.Config,
    expected: []const u8,
) !window.CandidateMeasurement {
    // Candidate executables and their argument/result handles are scoped to
    // one measurement. This keeps the tuning sweep from retaining private
    // graph pools for every rejected geometry before serving replay starts.
    var executable = try platform.compileFn(allocator, io, run, .{ tensors, candidate }, .{});
    defer executable.deinit();
    var arguments = try executable.args(allocator);
    defer arguments.deinit(allocator);
    var results = try executable.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ buffers, candidate });
    executable.call(arguments, &results);
    var output = results.get(zml.Buffer);
    try output.await(io);
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    output.deinit();
    const stats = try compareF16(actual.bytes, expected, candidate.rank8_enabled == 0);
    const median_ns = try window.benchmarkExecutable(
        allocator,
        io,
        &executable,
        arguments,
        &results,
        .{ .warmup_calls = 1, .iterations = 3 },
    );
    return .{
        .median_ns = median_ns,
        .mean_absolute_error = stats.mean,
        .max_absolute_error = stats.max,
        .accepted = stats.accepted,
        .arithmetic_signature = if (candidate.recovery_kernel == 1 or candidate.recovery_projection == 2)
            .unverified
        else
            .reference_fp32_v1,
    };
}

const TuningEntry = struct {
    rank8_enabled: u32,
    candidate_index: u32,
    config: window.Config,
    median_ns: u64,
    mean_absolute_error: f32,
    max_absolute_error: f32,
    accepted: bool,
    arithmetic_signature: window.ArithmeticSignature,
    recovery_pair: ?window.RecoveryPairMeasurement = null,
};

const TuningReport = struct {
    schema_version: u32 = 2,
    fixture: []const u8,
    target: []const u8,
    device_kind: []const u8,
    device_debug: []const u8,
    device_hardware_id: i32,
    artifact_payload_sha256: ?[]const u8 = null,
    max_recovery_overhead_percent: ?f64,
    selection_policy_off: window.QualityMode = .fast,
    selection_policy_on: window.QualityMode = .quality,
    selected_off: ?u32 = null,
    selected_on: ?u32 = null,
    entries: []const TuningEntry,
};

fn writeTuningReport(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    report: TuningReport,
) !void {
    var encoded: std.Io.Writer.Allocating = .init(allocator);
    defer encoded.deinit();
    try encoded.writer.print("{f}\n", .{std.json.fmt(report, .{ .emit_null_optional_fields = true })});
    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, encoded.written(), 0);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, struct {
        fixture: []const u8,
        /// Override the optional budget embedded in the fixture manifest.
        max_recovery_overhead_percent: ?f64 = null,
        /// Write the complete correctness-gated candidate sweep and selected
        /// geometries as a reusable report. Tuning and graph preparation
        /// still happen before serving capture; this file is telemetry only.
        tuning_output: ?[]const u8 = null,
        /// Override the fixture's requested correction-on arithmetic policy.
        /// The default is the manifest value, then quality.
        quality_mode: ?window.QualityMode = null,
    });
    var directory = try std.Io.Dir.cwd().openDir(io, args.fixture, .{});
    defer directory.close(io);
    const metadata = try directory.readFileAlloc(io, "manifest.json", allocator, .limited(1 << 20));
    defer allocator.free(metadata);
    const manifest = try std.json.parseFromSlice(std.json.Value, allocator, metadata, .{});
    defer manifest.deinit();
    try window.validateFixtureRecoveryManifest(manifest.value.object);
    const manifest_quality: ?window.QualityMode = if (manifest.value.object.get("quality_mode")) |value|
        switch (value) {
            .string => |name| std.meta.stringToEnum(window.QualityMode, name) orelse return error.InvalidQualityMode,
            else => return error.InvalidQualityMode,
        }
    else
        null;
    const quality_mode = args.quality_mode orelse manifest_quality orelse .quality;
    const parsed = try std.json.parseFromValue(window.Config, allocator, manifest.value.object.get("config").?, .{});
    defer parsed.deinit();
    var config = parsed.value;
    // A fixture may declare a promotion budget for the marginal rank8 cost.
    // Omit it for report-only verification; when present, the on-state
    // selector requires a matched off/on measurement for the same geometry.
    const manifest_budget: ?f64 = if (manifest.value.object.get("max_recovery_overhead_percent")) |value|
        switch (value) {
            .integer => |number| @as(f64, @floatFromInt(number)),
            .float => |number| number,
            else => return error.InvalidRecoveryBudget,
        }
    else
        null;
    const recovery_budget = args.max_recovery_overhead_percent orelse manifest_budget;
    if (recovery_budget) |budget| {
        if (!std.math.isFinite(budget) or budget < 0) return error.InvalidRecoveryBudget;
    }
    const artifact_payload_sha256: ?[]const u8 = if (manifest.value.object.get("payload_sha256")) |value|
        switch (value) {
            .string => |digest| digest,
            else => return error.InvalidArtifactPayloadHash,
        }
    else
        null;
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.5 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.CudaPlatformRequired;
    std.log.info("window verification platform: {f}", .{platform.fmtVerbose()});
    const paths = manifest.value.object.get("libraries").?.array.items;
    var runtime = try window.Runtime.init(.{ paths[0].string, paths[1].string, paths[2].string }, platform);
    defer runtime.deinit();
    var tensors: window.Input = undefined;
    var buffers: zml.Bufferized(window.Input) = undefined;
    inline for (@typeInfo(window.Input).@"struct".fields) |field| {
        const name = field.name;
        const bytes = try directory.readFileAlloc(io, name ++ ".bin", allocator, .limited(1 << 30));
        defer allocator.free(bytes);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        const expected_hash = manifest.value.object.get("files").?.object.get(name).?.object.get("sha256").?.string;
        if (!std.mem.eql(u8, &hex, expected_hash)) return error.FixtureHashMismatch;
        const shape: zml.Shape = if (comptime std.mem.eql(u8, name, "x"))
            .init(.{ config.m, config.k }, .f16)
        else if (comptime std.mem.eql(u8, name, "window"))
            .init(.{ config.k * config.n / 256, 4 * config.transition_bits }, .i32)
        else if (comptime std.mem.eql(u8, name, "banks"))
            .init(.{config.k * config.n / 256}, .u8)
        else if (comptime std.mem.eql(u8, name, "levels"))
            .init(.{256}, .f16)
        else if (comptime std.mem.eql(u8, name, "su"))
            .init(.{config.k}, .f16)
        else if (comptime std.mem.eql(u8, name, "rank8_a"))
            .init(.{ config.k, 8 }, .f16)
        else if (comptime std.mem.eql(u8, name, "rank8_b"))
            .init(.{ 8, config.n }, .f16)
        else
            .init(.{bytes.len / 2}, .f16);
        @field(tensors, name) = .fromShape(shape);
        @field(buffers, name) = try .fromBytes(io, platform, shape, .replicated, bytes);
    }
    defer inline for (@typeInfo(window.Input).@"struct".fields) |field| @field(buffers, field.name).deinit();
    var off_medians: [window.max_candidate_count]u64 = @splat(0);
    var report_entries: [window.max_candidate_count * 2]TuningEntry = undefined;
    var report_entry_count: usize = 0;
    var selected_off: ?u32 = null;
    var selected_on: ?u32 = null;
    for (0..2) |enabled| {
        config.rank8_enabled = @intCast(enabled);
        const expected = try directory.readFileAlloc(
            io,
            if (enabled == 1) "expected_on.bin" else "expected_off.bin",
            allocator,
            .limited(1 << 30),
        );
        defer allocator.free(expected);

        // Compile, warm, correctness-check and measure every BM/BN candidate
        // before selecting the serving geometry. This is deliberately outside
        // command-buffer capture; the selected executable is prepared again
        // below for the replay check.
        var candidates: [window.max_candidate_count]window.Config = undefined;
        const candidate_count = window.enumerateCandidatesForShape(config, &candidates);
        if (candidate_count == 0) return error.NoWindowCandidates;
        var measurements: [window.max_candidate_count]window.CandidateMeasurement = undefined;
        for (candidates[0..candidate_count], 0..) |candidate, index| {
            measurements[index] = try measureCandidate(
                allocator,
                io,
                platform,
                tensors,
                buffers,
                candidate,
                expected,
            );
            if (enabled == 0) {
                off_medians[index] = measurements[index].median_ns;
            } else if (off_medians[index] != 0) {
                measurements[index].recovery_pair =
                    try window.recoveryPairFromMedians(off_medians[index], measurements[index].median_ns);
            }
            report_entries[report_entry_count] = .{
                .rank8_enabled = @intCast(enabled),
                .candidate_index = @intCast(index),
                .config = candidate,
                .median_ns = measurements[index].median_ns,
                .mean_absolute_error = measurements[index].mean_absolute_error,
                .max_absolute_error = measurements[index].max_absolute_error,
                .accepted = measurements[index].accepted,
                .arithmetic_signature = measurements[index].arithmetic_signature,
                .recovery_pair = measurements[index].recovery_pair,
            };
            report_entry_count += 1;
            std.log.info(
                "ZML candidate rank8={d} index={d} algorithm={d} BM={d} BN={d} median_ns={d} accepted={}",
                .{ enabled, index, candidate.algorithm, candidate.block_m, candidate.block_n, measurements[index].median_ns, measurements[index].accepted },
            );
            if (enabled == 1) {
                if (measurements[index].recovery_pair) |pair| {
                    std.log.info(
                        "ZML candidate index={d} matched recovery overhead_ns={d} overhead_percent={d:.3}",
                        .{ index, pair.overhead_ns, pair.overhead_percent },
                    );
                } else {
                    std.log.warn("ZML candidate index={d} has no matched correction-off timing", .{index});
                }
            }
        }
        const winner = if (enabled == 1)
            try window.selectFastestWithPolicy(
                candidates[0..candidate_count],
                measurements[0..candidate_count],
                recovery_budget,
                quality_mode,
            )
        else
            try window.selectFastestWithPolicy(
                candidates[0..candidate_count],
                measurements[0..candidate_count],
                null,
                .fast,
            );
        const selected = winner.config;
        if (enabled == 0) selected_off = @intCast(winner.candidate_index) else selected_on = @intCast(winner.candidate_index);
        var executable = try platform.compileFn(allocator, io, run, .{ tensors, selected }, .{});
        defer executable.deinit();
        var arguments = try executable.args(allocator);
        defer arguments.deinit(allocator);
        var results = try executable.results(allocator);
        defer results.deinit(allocator);
        arguments.set(.{ buffers, selected });
        // The first eager call prepares and retains the native graph. The
        // second call exercises ordinary replay through the same handle and
        // catches accidental per-call graph creation or pointer churn.
        for (0..2) |_| executable.call(arguments, &results);
        var output = results.get(zml.Buffer);
        defer output.deinit();
        const actual = try output.toSliceAlloc(allocator, io);
        defer actual.free(allocator);
        const selected_stats = try compareF16(actual.bytes, expected, enabled == 0);
        if (!selected_stats.accepted) return error.WindowOutputOutsideNumericalGate;
        std.log.info(
            "ZML window rank8={d}: selected index={d} median_ns={d} mean_abs={d:.6} max_abs={d:.6} {d}x{d} output",
            .{ enabled, winner.candidate_index, winner.median_ns, selected_stats.mean, selected_stats.max, config.m, config.n },
        );
    }
    if (args.tuning_output) |path| {
        try writeTuningReport(allocator, io, path, .{
            .fixture = args.fixture,
            .target = @tagName(platform.target),
            .device_kind = platform.devices[0].kind(),
            .device_debug = platform.devices[0].debugString(),
            .device_hardware_id = platform.devices[0].localHardwareId(),
            .artifact_payload_sha256 = artifact_payload_sha256,
            .max_recovery_overhead_percent = recovery_budget,
            .selection_policy_off = .fast,
            .selection_policy_on = quality_mode,
            .selected_off = selected_off,
            .selected_on = selected_on,
            .entries = report_entries[0..report_entry_count],
        });
        std.log.info("wrote graph-safe ZML tuning report: {s}", .{path});
    }
}
