// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const zml = @import("zml");

pub const Config = extern struct {
    abi_version: u32 = 3,
    struct_bytes: u32 = @sizeOf(Config),
    m: u32,
    k: u32,
    n: u32,
    transition_bits: u32,
    bank_alt_id: u32,
    algorithm: u32,
    block_m: u32 = 0,
    block_n: u32 = 0,
    block_k: u32 = 256,
    warp_groups: u32 = 0,
    pipeline_stages: u32 = 2,
    split_k: u32 = 1,
    min_m: u32 = 1,
    max_m: u32 = 8192,
    input_hadamard: u32 = 1,
    output_hadamard: u32 = 1,
    rank8_enabled: u32 = 0,
};
const NativeBuffer = extern struct { data: ?*anyopaque, bytes: u64 };
const NativeGraphCreate = *const fn (
    [*]const NativeBuffer,
    *const Config,
    ?*anyopaque,
    *?*anyopaque,
    [*]u8,
    u64,
) callconv(.c) c_int;
const NativeGraphRun = *const fn (
    ?*anyopaque,
    ?*anyopaque,
    [*]u8,
    u64,
) callconv(.c) c_int;
const NativeGraphDestroy = *const fn (
    ?*anyopaque,
    [*]u8,
    u64,
) callconv(.c) c_int;

const GraphKey = struct {
    buffers: [10]usize,
    stream: usize,
    config: Config,
};

const GraphKeyContext = struct {
    pub fn hash(_: @This(), key: GraphKey) u64 {
        var result: u64 = 0;
        for (key.buffers) |pointer| {
            result = std.hash.Wyhash.hash(result, std.mem.asBytes(&pointer));
        }
        result = std.hash.Wyhash.hash(result, std.mem.asBytes(&key.stream));
        return std.hash.Wyhash.hash(result, std.mem.asBytes(&key.config));
    }

    pub fn eql(_: @This(), left: GraphKey, right: GraphKey) bool {
        return std.mem.eql(usize, &left.buffers, &right.buffers) and left.stream == right.stream and std.mem.eql(u8, std.mem.asBytes(&left.config), std.mem.asBytes(&right.config));
    }
};

const GraphMap = std.HashMap(
    GraphKey,
    ?*anyopaque,
    GraphKeyContext,
    std.hash_map.default_max_load_percentage,
);

const GraphLock = struct {
    held: std.atomic.Value(bool) = .init(false),

    fn lock(self: *@This()) void {
        while (self.held.swap(true, .acquire)) {}
    }

    fn unlock(self: *@This()) void {
        self.held.store(false, .release);
    }
};

var native_graph_create: ?NativeGraphCreate = null;
var native_graph_run: ?NativeGraphRun = null;
var native_graph_destroy: ?NativeGraphDestroy = null;
var graph_mutex: GraphLock = .{};
var graph_handles: ?GraphMap = null;

fn pointerValue(pointer: ?*anyopaque) usize {
    return if (pointer) |value| @intFromPtr(value) else 0;
}

pub const Runtime = struct {
    libraries: [3]std.DynLib,

    // Load existing QVQ CUDA, Hopper WGMMA, then the native window ABI library.
    // Keep this runtime alive until every executable and captured graph using
    // it has finished. Prepared handles own a native allocator pool and are
    // released by deinit after the owning stream is synchronized.
    pub fn init(paths: [3][]const u8, platform: *const zml.Platform) !Runtime {
        if (native_graph_create != null) return error.AlreadyInitialized;
        var libraries: [3]std.DynLib = undefined;
        var loaded: usize = 0;
        errdefer for (libraries[0..loaded]) |*library| library.close();
        for (paths, 0..) |path, i| {
            libraries[i] = try std.DynLib.open(path);
            loaded += 1;
        }
        native_graph_create = libraries[2].lookup(NativeGraphCreate, "qvq_p32_window_graph_create") orelse
            return error.MissingNativeWindowGraphSymbol;
        native_graph_run = libraries[2].lookup(NativeGraphRun, "qvq_p32_window_graph_run") orelse
            return error.MissingNativeWindowGraphSymbol;
        native_graph_destroy = libraries[2].lookup(NativeGraphDestroy, "qvq_p32_window_graph_destroy") orelse
            return error.MissingNativeWindowGraphSymbol;
        errdefer {
            native_graph_create = null;
            native_graph_run = null;
            native_graph_destroy = null;
        }
        try platform.registerFfi(.{
            .name = "qvq_p32_window_linear",
            .handler = handler,
            // The handler submits only a prepared graph. Its first eager call
            // prepares the graph outside capture; capture then reuses the
            // retained graph or inserts it as a child node.
            .traits = .{ .command_buffer_compatible = true },
        });
        graph_mutex.lock();
        graph_handles = GraphMap.init(std.heap.c_allocator);
        graph_mutex.unlock();
        return .{ .libraries = libraries };
    }

    pub fn deinit(self: *Runtime) void {
        graph_mutex.lock();
        if (graph_handles) |*handles| {
            var iterator = handles.iterator();
            var message: [4096]u8 = @splat(0);
            while (iterator.next()) |entry| {
                _ = native_graph_destroy.?(
                    entry.value_ptr.*,
                    &message,
                    message.len,
                );
            }
            handles.deinit();
            graph_handles = null;
        }
        graph_mutex.unlock();
        native_graph_create = null;
        native_graph_run = null;
        native_graph_destroy = null;
        // Torch dispatcher registrations are removed by the library destructors.
        var i: usize = self.libraries.len;
        while (i > 0) {
            i -= 1;
            self.libraries[i].close();
        }
    }
};

pub const Input = struct {
    x: zml.Tensor,
    window: zml.Tensor,
    banks: zml.Tensor,
    levels: zml.Tensor,
    su: zml.Tensor,
    sv: zml.Tensor,
    bias: zml.Tensor,
    rank8_a: zml.Tensor,
    rank8_b: zml.Tensor,
};

/// Native Hopper configurations that ZML may compile and benchmark for one
/// shape. The list intentionally retains every supported BM/BN choice: the
/// winning geometry is shape-, device- and correction-state dependent.
pub const max_candidate_count: usize = 7;

/// Result of one correctness-gated ZML candidate measurement. The executable
/// and timing loop belong to the caller so it can use its own PJRT client and
/// stream. Measurements must be collected before the final executable is
/// captured; this selector is allocation-free and safe to call during graph
/// planning, but must not be called from a captured custom-call handler.
pub const CandidateMeasurement = struct {
    median_ns: u64,
    mean_absolute_error: f32,
    max_absolute_error: f32,
    accepted: bool,
};

pub const TuningResult = struct {
    config: Config,
    median_ns: u64,
    candidate_index: usize,
};

/// Select the fastest locally correct candidate in stable enumeration order.
/// The error gate matches the Python/native QvQ contract: a finite output,
/// mean absolute error <= 2e-3 and max absolute error <= 3/64. Invalid or
/// zero-duration samples are rejected rather than silently becoming winners.
/// ZML callers compile, warm and benchmark each `linear` candidate outside
/// capture, then pass the resulting samples here before compiling the graph
/// that will serve requests.
pub fn selectFastest(
    candidates: []const Config,
    measurements: []const CandidateMeasurement,
) !TuningResult {
    if (candidates.len == 0) return error.NoCandidates;
    if (candidates.len != measurements.len) return error.MeasurementCountMismatch;
    var selected: ?TuningResult = null;
    for (candidates, measurements, 0..) |candidate, measurement, index| {
        if (!measurement.accepted or measurement.median_ns == 0 or
            !std.math.isFinite(measurement.mean_absolute_error) or
            !std.math.isFinite(measurement.max_absolute_error) or
            measurement.mean_absolute_error > 2e-3 or
            measurement.max_absolute_error > 3.0 / 64.0) continue;
        if (selected == null or measurement.median_ns < selected.?.median_ns) {
            selected = .{
                .config = candidate,
                .median_ns = measurement.median_ns,
                .candidate_index = index,
            };
        }
    }
    return selected orelse error.NoViableCandidate;
}

pub fn enumerateCandidates(base: Config, output: []Config) usize {
    if (output.len < max_candidate_count) return 0;
    var count: usize = 0;
    var m16 = base;
    m16.algorithm = 1;
    m16.block_m = 0;
    m16.block_n = 0;
    m16.warp_groups = 0;
    output[count] = m16;
    count += 1;
    for ([_]u32{ 32, 64, 128 }) |block_m| {
        for ([_]u32{ 64, 128 }) |block_n| {
            var candidate = base;
            candidate.algorithm = 2;
            candidate.block_m = block_m;
            candidate.block_n = block_n;
            // Zero lets the native launcher select its normal warp-group
            // policy; callers may override it when their tuner supports it.
            candidate.warp_groups = 0;
            output[count] = candidate;
            count += 1;
        }
    }
    return count;
}

// Config is explicit compiler data: ZML may enumerate supported BM/BN/M
// candidates before lowering. rank8_enabled is resolved by quantizer metadata
// and requested quality mode, not chosen merely by latency.
pub fn linear(input: Input, config: Config) zml.Tensor {
    std.debug.assert(config.abi_version == 3 and config.struct_bytes == @sizeOf(Config));
    std.debug.assert(input.x.rank() == 2);
    std.debug.assert(input.x.dim(0) == config.m and input.x.dim(1) == config.k);
    inline for (@typeInfo(Input).@"struct".fields) |field| {
        const expected: zml.DataType = if (comptime std.mem.eql(u8, field.name, "window"))
            .i32
        else if (comptime std.mem.eql(u8, field.name, "banks"))
            .u8
        else
            .f16;
        std.debug.assert(@field(input, field.name).dtype() == expected);
    }
    // Ordinary typedCustomCall keeps replication semantics; no TP ownership
    // claim is made until the sharding contract is separately validated.
    return zml.ops.typedCustomCall(
        "qvq_p32_window_linear",
        .{ .has_side_effect = false },
        input,
        .{ .y = zml.Shape.init(.{ config.m, config.n }, .f16) },
        config,
    ).y;
}

fn handler(frame: *zml.pjrt.ffi.CallFrame) callconv(.c) ?*zml.pjrt.ffi.Error {
    if (frame.registeringHook()) return null;
    if (native_graph_create == null or native_graph_run == null or native_graph_destroy == null)
        return zml.pjrt.ffi.Error.create(frame.api, .failed_precondition, "native window runtime is not initialized");
    const inputs = frame.args.buffers();
    const outputs = frame.results.buffers();
    if (inputs.len != 9 or outputs.len != 1)
        return zml.pjrt.ffi.Error.create(frame.api, .invalid_argument, "invalid window buffer arity");
    var config: Config = undefined;
    inline for (@typeInfo(Config).@"struct".fields) |field| {
        const attr = frame.attrs.getByName(.scalar, field.name) orelse
            return zml.pjrt.ffi.Error.create(frame.api, .invalid_argument, "missing window configuration attribute");
        @field(config, field.name) = attr.get(u32);
    }
    var buffers: [10]NativeBuffer = undefined;
    for (inputs, 0..) |value, i| {
        const buffer = zml.pjrtx.CustomCallBuffer.fromPjrt(value);
        const bytes = buffer.shape.byteSize();
        buffers[i] = .{ .data = if (bytes == 0) null else buffer.ptr, .bytes = bytes };
    }
    const output = zml.pjrtx.CustomCallBuffer.fromPjrt(outputs[0]);
    buffers[9] = .{ .data = output.ptr, .bytes = output.shape.byteSize() };
    const stream: ?*anyopaque = @ptrCast(frame.api.stream(frame.ctx));
    var key: GraphKey = .{
        .buffers = undefined,
        .stream = pointerValue(stream),
        .config = config,
    };
    for (buffers, 0..) |buffer, i| key.buffers[i] = pointerValue(buffer.data);

    // A first eager call prepares the retained native graph. Once the backend
    // starts capture, a missing key is rejected by graph_create before any
    // allocation or event creation, so callers must warm each executable and
    // buffer set before capturing it.
    graph_mutex.lock();
    defer graph_mutex.unlock();
    const handles = if (graph_handles) |*value| value else return zml.pjrt.ffi.Error.create(frame.api, .failed_precondition, "native window graph registry is not initialized");
    var handle: ?*anyopaque = null;
    if (handles.getPtr(key)) |entry| {
        handle = entry.*;
    } else {
        var message: [4096]u8 = @splat(0);
        const status = native_graph_create.?(
            &buffers,
            &config,
            stream,
            &handle,
            &message,
            message.len,
        );
        if (status != 0 or handle == null)
            return zml.pjrt.ffi.Error.create(frame.api, .internal, std.mem.sliceTo(&message, 0));
        handles.put(key, handle) catch {
            var cleanup_message: [4096]u8 = @splat(0);
            _ = native_graph_destroy.?(handle, &cleanup_message, cleanup_message.len);
            return zml.pjrt.ffi.Error.create(frame.api, .resource_exhausted, "unable to retain native window graph handle");
        };
    }
    var message: [4096]u8 = @splat(0);
    const status = native_graph_run.?(
        handle,
        stream,
        &message,
        message.len,
    );
    if (status != 0)
        return zml.pjrt.ffi.Error.create(frame.api, .internal, std.mem.sliceTo(&message, 0));
    return null;
}

test "native window ABI layout" {
    try std.testing.expectEqual(@as(usize, 76), @sizeOf(Config));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(NativeBuffer));
    var candidates: [max_candidate_count]Config = undefined;
    const count = enumerateCandidates(.{ .m = 33, .k = 2048, .n = 2048, .transition_bits = 4, .bank_alt_id = 2, .algorithm = 2 }, &candidates);
    try std.testing.expectEqual(max_candidate_count, count);
    try std.testing.expectEqual(@as(u32, 1), candidates[0].algorithm);
    try std.testing.expectEqual(@as(u32, 2), candidates[1].algorithm);
    try std.testing.expectEqual(@as(u32, 32), candidates[1].block_m);
    try std.testing.expectEqual(@as(u32, 64), candidates[1].block_n);
    var measurements: [max_candidate_count]CandidateMeasurement = @splat(.{
        .median_ns = 100,
        .mean_absolute_error = 0,
        .max_absolute_error = 0,
        .accepted = true,
    });
    measurements[3].median_ns = 20;
    const winner = try selectFastest(candidates[0..count], &measurements);
    try std.testing.expectEqual(@as(usize, 3), winner.candidate_index);
    try std.testing.expectEqual(@as(u64, 20), winner.median_ns);
    measurements[3].accepted = false;
    const fallback = try selectFastest(candidates[0..count], &measurements);
    try std.testing.expectEqual(@as(usize, 0), fallback.candidate_index);
    for (&measurements) |*measurement| measurement.accepted = false;
    measurements[0].max_absolute_error = 1;
    try std.testing.expectError(error.NoViableCandidate, selectFastest(candidates[0..count], &measurements));
    std.testing.refAllDecls(@This());
}
