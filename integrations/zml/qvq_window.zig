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
    // Optional rank8 implementation policy passed through the native ABI:
    // recovery_kernel 0 = separate reference, 1 = fused epilogue,
    // 2 = fully fused project-output epilogue;
    // recovery_projection 0 = separate reference, 1 = concurrent reference,
    // 2 = concurrent Tensor Core, 3 = input-fused producer,
    // 4 = project-output fused producer (2..4 are unverified/fast-only).
    // Unsupported combinations fail closed in the native consumer rather than
    // silently degrading.
    recovery_kernel: u32 = 0,
    recovery_projection: u32 = 0,
};

/// Quality policy is deliberately separate from launch geometry.  A latency
/// winner cannot silently change the arithmetic contract of a quality graph.
pub const QualityMode = enum {
    fast,
    balanced,
    quality,
};

/// Arithmetic signatures are assigned by the producer after numerical
/// certification.  Unknown signatures remain visible and are only eligible
/// for the explicitly permissive fast mode.
pub const ArithmeticSignature = enum(u8) {
    reference_fp32_v1 = 1,
    certified_tensor_core_v1 = 2,
    unverified = 255,
};

fn arithmeticAllowed(mode: QualityMode, signature: ArithmeticSignature) bool {
    return switch (mode) {
        .fast => true,
        .balanced => signature == .reference_fp32_v1 or signature == .certified_tensor_core_v1,
        .quality => signature == .reference_fp32_v1,
    };
}

/// Validate the complete external ABI policy before lowering or replay.
/// Keeping this pure lets ZML candidate generation and host-side tests reject
/// unsupported geometry without allocating device state.  The native bridge
/// repeats the checks at its own boundary.
pub fn configValid(config: Config) bool {
    if (config.abi_version != 3 or config.struct_bytes != @sizeOf(Config) or
        config.m == 0 or config.m > 8192 or config.k == 0 or config.n == 0 or
        config.transition_bits < 4 or config.transition_bits > 7 or
        config.bank_alt_id < 1 or config.bank_alt_id > 3 or
        config.min_m == 0 or config.min_m > config.m or
        config.max_m < config.m or config.max_m > 8192 or
        config.block_k != 256 or config.pipeline_stages != 2 or
        config.split_k != 1 or config.input_hadamard > 1 or
        config.output_hadamard > 1 or config.rank8_enabled > 1 or
        config.recovery_kernel > 2 or config.recovery_projection > 4)
        return false;
    if (config.rank8_enabled != 0) {
        if (config.recovery_kernel != 0 and config.output_hadamard != 0 and
            (config.n < 16 or (config.n & (config.n - 1)) != 0))
            return false;
        if (config.recovery_projection == 4 and
            (config.recovery_kernel == 0 or config.n > 16384 or
             (config.output_hadamard != 0 and
              (config.n < 16 or (config.n & (config.n - 1)) != 0))))
            return false;
        if (config.recovery_kernel == 2 and config.recovery_projection != 4)
            return false;
    }
    if (config.algorithm == 1) {
        if (config.block_m != 0 or config.block_n != 0 or config.warp_groups != 0)
            return false;
    } else if (config.algorithm == 2) {
        if (config.block_m != 0 and (config.block_m != 32 and config.block_m != 64 and config.block_m != 128))
            return false;
        if (config.block_n != 0 and (config.block_n != 64 and config.block_n != 128))
            return false;
        if (config.warp_groups != 0 and
            (config.block_n == 0 or config.warp_groups != config.block_n / 64))
            return false;
        if (config.block_m == 0 or config.block_n == 0) {
            if (config.warp_groups != 0) return false;
        }
    } else {
        return false;
    }
    return true;
}
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
    buffer_bytes: [10]u64,
    buffer_types: [10]u8,
    device_ordinal: usize,
    stream: usize,
    config: Config,
};

const GraphKeyContext = struct {
    pub fn hash(_: @This(), key: GraphKey) u64 {
        var result: u64 = 0;
        for (key.buffers) |pointer| {
            result = std.hash.Wyhash.hash(result, std.mem.asBytes(&pointer));
        }
        for (key.buffer_bytes) |bytes| {
            result = std.hash.Wyhash.hash(result, std.mem.asBytes(&bytes));
        }
        for (key.buffer_types) |dtype| {
            result = std.hash.Wyhash.hash(result, std.mem.asBytes(&dtype));
        }
        result = std.hash.Wyhash.hash(result, std.mem.asBytes(&key.device_ordinal));
        result = std.hash.Wyhash.hash(result, std.mem.asBytes(&key.stream));
        return std.hash.Wyhash.hash(result, std.mem.asBytes(&key.config));
    }

    pub fn eql(_: @This(), left: GraphKey, right: GraphKey) bool {
        return std.mem.eql(usize, &left.buffers, &right.buffers) and
            std.mem.eql(u64, &left.buffer_bytes, &right.buffer_bytes) and
            std.mem.eql(u8, &left.buffer_types, &right.buffer_types) and
            left.device_ordinal == right.device_ordinal and
            left.stream == right.stream and
            std.mem.eql(u8, std.mem.asBytes(&left.config), std.mem.asBytes(&right.config));
    }
};

const GraphLock = struct {
    held: std.atomic.Mutex = .unlocked,

    fn tryLock(self: *@This()) bool {
        return self.held.tryLock();
    }

    fn lock(self: *@This()) void {
        while (!self.held.tryLock()) {
            // The registry is never held over native creation/replay. Yield
            // rather than burning a CPU in a tight busy-spin when two PJRT
            // threads insert or evict different executable handles.
            std.Thread.yield() catch {};
        }
    }

    fn unlock(self: *@This()) void {
        self.held.unlock();
    }
};

const GraphEntry = struct {
    handle: ?*anyopaque,
    last_used: u64,
    run_lock: GraphLock = .{},
};

const GraphMap = std.HashMap(
    GraphKey,
    *GraphEntry,
    GraphKeyContext,
    std.hash_map.default_max_load_percentage,
);
const max_graph_handles: usize = 256;

var native_graph_create: ?NativeGraphCreate = null;
var native_graph_run: ?NativeGraphRun = null;
var native_graph_destroy: ?NativeGraphDestroy = null;
var graph_mutex: GraphLock = .{};
var graph_handles: ?GraphMap = null;
var graph_use_counter: u64 = 0;

fn pointerValue(pointer: ?*anyopaque) usize {
    return if (pointer) |value| @intFromPtr(value) else 0;
}

fn destroyNativeGraph(handle: ?*anyopaque) void {
    if (handle == null) return;
    var message: [4096]u8 = @splat(0);
    _ = native_graph_destroy.?(handle, &message, message.len);
}

pub const Runtime = struct {
    libraries: [3]std.DynLib,

    // Load existing QVQ CUDA, Hopper WGMMA, then the native window ABI library.
    // Keep this runtime alive until every executable and captured graph using
    // it has finished. Prepared handles own a native allocator pool and are
    // released by deinit after the owning stream is synchronized.
    //
    // Loading is intentionally separate from FFI registration. LibTorch and
    // ZML both carry CUDA 13 shared objects; loading the QVQ libraries before
    // PJRT initializes its CUDA runtime prevents the dynamic loader from
    // binding LibTorch against PJRT's private CUDA copy.
    pub fn load(paths: [3][]const u8) !Runtime {
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
        graph_mutex.lock();
        graph_handles = GraphMap.init(std.heap.c_allocator);
        graph_use_counter = 0;
        graph_mutex.unlock();
        return .{ .libraries = libraries };
    }

    pub fn register(self: *Runtime, platform: *const zml.Platform) !void {
        _ = self;
        try platform.registerFfi(.{
            .name = "qvq_p32_window_linear",
            .handler = handler,
            // The handler submits only a prepared graph. Its first eager call
            // prepares the graph outside capture; capture then reuses the
            // retained graph or inserts it as a child node.
            .traits = .{ .command_buffer_compatible = true },
        });
    }

    pub fn deinit(self: *Runtime) void {
        // Detach entries while holding the map lock, then destroy native
        // graphs after unlocking. Native destruction synchronizes an owning
        // CUDA stream and must never block unrelated registry operations.
        var detached: [max_graph_handles]?*GraphEntry = @splat(null);
        var detached_count: usize = 0;
        graph_mutex.lock();
        if (graph_handles) |*handles| {
            var iterator = handles.iterator();
            while (iterator.next()) |entry| {
                if (detached_count < detached.len) {
                    detached[detached_count] = entry.value_ptr.*;
                    detached_count += 1;
                }
            }
            handles.deinit();
            graph_handles = null;
        }
        graph_use_counter = 0;
        graph_mutex.unlock();
        for (detached[0..detached_count]) |entry| {
            entry.?.run_lock.lock();
            destroyNativeGraph(entry.?.handle);
            entry.?.run_lock.unlock();
            std.heap.c_allocator.destroy(entry.?);
        }
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

pub const ArtifactPayload = struct {
    window: zml.Buffer,
    banks: zml.Buffer,
    levels: zml.Buffer,
    su: zml.Buffer,
    sv: zml.Buffer,
    bias: zml.Buffer,
    rank8_a: zml.Buffer,
    rank8_b: zml.Buffer,

    pub fn deinit(self: *@This()) void {
        self.window.deinit();
        self.banks.deinit();
        self.levels.deinit();
        self.su.deinit();
        self.sv.deinit();
        self.bias.deinit();
        self.rank8_a.deinit();
        self.rank8_b.deinit();
    }
};

pub const Artifact = struct {
    config: Config,
    payload: ArtifactPayload,

    pub fn deinit(self: *@This()) void {
        self.payload.deinit();
    }

    /// Return compiler-side tensor shapes for a runtime activation tensor.
    pub fn inputTensors(self: *const @This(), x: zml.Tensor) Input {
        return .{
            .x = x,
            .window = .fromShape(self.payload.window.shape()),
            .banks = .fromShape(self.payload.banks.shape()),
            .levels = .fromShape(self.payload.levels.shape()),
            .su = .fromShape(self.payload.su.shape()),
            .sv = .fromShape(self.payload.sv.shape()),
            .bias = .fromShape(self.payload.bias.shape()),
            .rank8_a = .fromShape(self.payload.rank8_a.shape()),
            .rank8_b = .fromShape(self.payload.rank8_b.shape()),
        };
    }

    /// Bind a runtime activation buffer to the immutable artifact payload.
    pub fn arguments(self: *const @This(), x: zml.Buffer) zml.Bufferized(Input) {
        return .{
            .x = x,
            .window = self.payload.window,
            .banks = self.payload.banks,
            .levels = self.payload.levels,
            .su = self.payload.su,
            .sv = self.payload.sv,
            .bias = self.payload.bias,
            .rank8_a = self.payload.rank8_a,
            .rank8_b = self.payload.rank8_b,
        };
    }
};

pub const ArtifactOptions = struct {
    m: u32,
    algorithm: u32 = 1,
    block_m: u32 = 0,
    block_n: u32 = 0,
    block_k: u32 = 256,
    warp_groups: u32 = 0,
    pipeline_stages: u32 = 2,
    split_k: u32 = 1,
    min_m: u32 = 1,
    max_m: u32 = 8192,
    rank8_enabled: bool = false,
};

fn artifactObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |object| object,
        else => error.InvalidWindowArtifactManifest,
    };
}

fn artifactField(object: std.json.ObjectMap, name: []const u8) !std.json.Value {
    return object.get(name) orelse error.InvalidWindowArtifactManifest;
}

fn artifactString(value: std.json.Value) ![]const u8 {
    return switch (value) {
        .string => |string| string,
        else => error.InvalidWindowArtifactManifest,
    };
}

fn artifactU32(value: std.json.Value) !u32 {
    return switch (value) {
        .integer => |integer| if (integer >= 0 and integer <= std.math.maxInt(u32))
            @intCast(integer)
        else
            error.InvalidWindowArtifactManifest,
        .float => |float| if (float >= 0 and float <= std.math.maxInt(u32) and @trunc(float) == float)
            @intFromFloat(float)
        else
            error.InvalidWindowArtifactManifest,
        else => error.InvalidWindowArtifactManifest,
    };
}

fn artifactBool(value: std.json.Value) !bool {
    return switch (value) {
        .bool => |boolean| boolean,
        else => error.InvalidWindowArtifactManifest,
    };
}

fn artifactShapeEquals(value: std.json.Value, expected: []const u32) !void {
    const array = switch (value) {
        .array => |array| array,
        else => return error.InvalidWindowArtifactManifest,
    };
    if (array.items.len != expected.len) return error.InvalidWindowArtifactManifest;
    for (array.items, expected) |actual, wanted| {
        if (try artifactU32(actual) != wanted) return error.InvalidWindowArtifactManifest;
    }
}

fn artifactHashMatches(bytes: []const u8, expected: []const u8) !void {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &hex, expected)) return error.WindowArtifactHashMismatch;
}

const ArtifactSha256 = std.crypto.hash.sha2.Sha256;

fn artifactBindingAppend(hasher: *ArtifactSha256, value: []const u8) void {
    hasher.update(value);
    hasher.update(&[_]u8{0});
}

fn artifactBindingAppendFmt(
    hasher: *ArtifactSha256,
    comptime format: []const u8,
    args: anytype,
) !void {
    var buffer: [128]u8 = undefined;
    const value = try std.fmt.bufPrint(&buffer, format, args);
    artifactBindingAppend(hasher, value);
}

fn artifactBindingAppendJsonString(hasher: *ArtifactSha256, value: []const u8) !void {
    var buffer: [512]u8 = undefined;
    const encoded = try std.fmt.bufPrint(&buffer, "\"{s}\"", .{value});
    artifactBindingAppend(hasher, encoded);
}

fn artifactBindingUpdateFmt(
    hasher: *ArtifactSha256,
    comptime format: []const u8,
    args: anytype,
) !void {
    var buffer: [128]u8 = undefined;
    const value = try std.fmt.bufPrint(&buffer, format, args);
    hasher.update(value);
}

fn artifactBindingAppendShape(
    hasher: *ArtifactSha256,
    value: std.json.Value,
) !void {
    const array = switch (value) {
        .array => |array| array,
        else => return error.InvalidWindowArtifactManifest,
    };
    hasher.update("[");
    for (array.items, 0..) |item, index| {
        if (index != 0) hasher.update(",");
        try artifactBindingUpdateFmt(hasher, "{d}", .{try artifactU32(item)});
    }
    hasher.update("]");
}

fn artifactBindingAppendEntry(
    hasher: *ArtifactSha256,
    tensors: std.json.ObjectMap,
    name: []const u8,
) !void {
    const value = tensors.get(name) orelse return;
    const entry = try artifactObject(value);
    artifactBindingAppend(hasher, name);
    artifactBindingAppend(hasher, "dtype");
    try artifactBindingAppendJsonString(
        hasher,
        try artifactString(try artifactField(entry, "dtype")),
    );
    artifactBindingAppend(hasher, "shape");
    try artifactBindingAppendShape(hasher, try artifactField(entry, "shape"));
    hasher.update(&[_]u8{0});
    artifactBindingAppend(hasher, "bytes");
    try artifactBindingAppendFmt(hasher, "{d}", .{try artifactU32(try artifactField(entry, "bytes"))});
    artifactBindingAppend(hasher, "sha256");
    try artifactBindingAppendJsonString(
        hasher,
        try artifactString(try artifactField(entry, "sha256")),
    );
}

fn artifactBindingHex(metadata: std.json.ObjectMap, tensors: std.json.ObjectMap) ![64]u8 {
    var hasher = ArtifactSha256.init(.{});
    artifactBindingAppend(&hasher, "qvq_p32_window_artifact-binding-v1");
    artifactBindingAppend(&hasher, "bits");
    switch (try artifactField(metadata, "bits")) {
        .integer => |value| try artifactBindingAppendFmt(&hasher, "{d}", .{value}),
        .float => |value| try artifactBindingAppendFmt(&hasher, "{d}", .{value}),
        else => return error.InvalidWindowArtifactManifest,
    }
    artifactBindingAppend(&hasher, "codebook_version");
    try artifactBindingAppendJsonString(
        &hasher,
        try artifactString(try artifactField(metadata, "codebook_version")),
    );
    artifactBindingAppend(&hasher, "in_features");
    try artifactBindingAppendFmt(&hasher, "{d}", .{try artifactU32(try artifactField(metadata, "in_features"))});
    artifactBindingAppend(&hasher, "out_features");
    try artifactBindingAppendFmt(&hasher, "{d}", .{try artifactU32(try artifactField(metadata, "out_features"))});
    artifactBindingAppend(&hasher, "input_hadamard");
    artifactBindingAppend(&hasher, if (try artifactBool(try artifactField(metadata, "input_hadamard"))) "true" else "false");
    artifactBindingAppend(&hasher, "output_hadamard");
    artifactBindingAppend(&hasher, if (try artifactBool(try artifactField(metadata, "output_hadamard"))) "true" else "false");
    // Python's sorted() order is stable and these are the only tensor names
    // accepted by the native ABI. Unknown entries are rejected below.
    for ([_][]const u8{
        "SU", "SV", "bank_alt_id", "bank_ids", "bias", "levels", "rank8_A", "rank8_B", "window_words",
    }) |name| try artifactBindingAppendEntry(&hasher, tensors, name);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

// This mirrors qvq_rank8._digest exactly for the fixed tensor contract used by
// the native handoff.  The descriptor is the compact JSON tuple emitted by
// Python (name, torch dtype, shape), followed immediately by the tensor's
// little-endian file bytes.  Keep this validation in the loader rather than
// trusting a producer-supplied semantic hash; all work happens before device
// upload and therefore never enters a captured execution path.
fn artifactSemanticTensor(
    hasher: *ArtifactSha256,
    allocator: std.mem.Allocator,
    io: std.Io,
    directory: *std.Io.Dir,
    tensors: std.json.ObjectMap,
    manifest_name: []const u8,
    digest_name: []const u8,
) !void {
    const value = tensors.get(manifest_name) orelse return;
    const entry = try artifactObject(value);
    const dtype = try artifactString(try artifactField(entry, "dtype"));
    const shape = switch (try artifactField(entry, "shape")) {
        .array => |array| array,
        else => return error.InvalidWindowArtifactManifest,
    };
    var descriptor: [512]u8 = undefined;
    const descriptor_bytes = switch (shape.items.len) {
        1 => try std.fmt.bufPrint(
            &descriptor,
            "[\"{s}\", \"torch.{s}\", [{d}]]",
            .{ digest_name, dtype, try artifactU32(shape.items[0]) },
        ),
        2 => try std.fmt.bufPrint(
            &descriptor,
            "[\"{s}\", \"torch.{s}\", [{d}, {d}]]",
            .{ digest_name, dtype, try artifactU32(shape.items[0]), try artifactU32(shape.items[1]) },
        ),
        else => return error.InvalidWindowArtifactManifest,
    };
    hasher.update(descriptor_bytes);
    const file = try artifactString(try artifactField(entry, "file"));
    const bytes = try directory.readFileAlloc(io, file, allocator, .limited(1 << 30));
    defer allocator.free(bytes);
    hasher.update(bytes);
}

fn artifactSemanticHash(
    allocator: std.mem.Allocator,
    io: std.Io,
    directory: *std.Io.Dir,
    metadata: std.json.ObjectMap,
    tensors: std.json.ObjectMap,
    comptime recovery_factors: bool,
) ![64]u8 {
    var hasher = ArtifactSha256.init(.{});
    if (recovery_factors) {
        // Python hashes _digest({"A": A, "B": B}, {}), whose canonical
        // metadata prefix is the empty JSON object.
        hasher.update("{}");
        try artifactSemanticTensor(&hasher, allocator, io, directory, tensors, "rank8_A", "A");
        try artifactSemanticTensor(&hasher, allocator, io, directory, tensors, "rank8_B", "B");
    } else {
        var bits_buffer: [64]u8 = undefined;
        const bits = switch (try artifactField(metadata, "bits")) {
            .integer => |value| try std.fmt.bufPrint(&bits_buffer, "{d}", .{value}),
            .float => |value| try std.fmt.bufPrint(&bits_buffer, "{d}", .{value}),
            else => return error.InvalidWindowArtifactManifest,
        };
        const codebook_version = try artifactString(try artifactField(metadata, "codebook_version"));
        const input_hadamard = if (try artifactBool(try artifactField(metadata, "input_hadamard"))) "true" else "false";
        const output_hadamard = if (try artifactBool(try artifactField(metadata, "output_hadamard"))) "true" else "false";
        var metadata_buffer: [512]u8 = undefined;
        const metadata_bytes = try std.fmt.bufPrint(
            &metadata_buffer,
            "{{\"bits\":{s},\"codebook_version\":\"{s}\",\"in_features\":{d},\"input_hadamard\":{s},\"out_features\":{d},\"output_hadamard\":{s}}}",
            .{
                bits,
                codebook_version,
                try artifactU32(try artifactField(metadata, "in_features")),
                input_hadamard,
                try artifactU32(try artifactField(metadata, "out_features")),
                output_hadamard,
            },
        );
        hasher.update(metadata_bytes);
        // Python's sorted() order for the accepted base tensors is stable and
        // differs from the manifest's insertion order for uppercase names.
        for ([_][]const u8{
            "SU", "SV", "bank_alt_id", "bank_ids", "bias", "levels", "window_words",
        }) |name| try artifactSemanticTensor(&hasher, allocator, io, directory, tensors, name, name);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

fn artifactValidateTensorNames(tensors: std.json.ObjectMap) !void {
    var iterator = tensors.iterator();
    while (iterator.next()) |item| {
        const name = item.key_ptr.*;
        if (!(std.mem.eql(u8, name, "SU") or std.mem.eql(u8, name, "SV") or
            std.mem.eql(u8, name, "bank_alt_id") or std.mem.eql(u8, name, "bank_ids") or
            std.mem.eql(u8, name, "bias") or std.mem.eql(u8, name, "levels") or
            std.mem.eql(u8, name, "rank8_A") or std.mem.eql(u8, name, "rank8_B") or
            std.mem.eql(u8, name, "window_words")))
            return error.InvalidWindowArtifactManifest;
    }
}

fn artifactTensor(
    allocator: std.mem.Allocator,
    io: std.Io,
    directory: *std.Io.Dir,
    tensors: std.json.ObjectMap,
    name: []const u8,
    dtype: []const u8,
    shape: []const u32,
    platform: *const zml.Platform,
) !zml.Buffer {
    const entry = try artifactObject(tensors.get(name) orelse return error.InvalidWindowArtifactManifest);
    const file = try artifactString(try artifactField(entry, "file"));
    var expected_file: [64]u8 = undefined;
    const expected_file_slice = try std.fmt.bufPrint(&expected_file, "{s}.bin", .{name});
    if (!std.mem.eql(u8, file, expected_file_slice)) return error.InvalidWindowArtifactManifest;
    if (!std.mem.eql(u8, try artifactString(try artifactField(entry, "dtype")), dtype))
        return error.InvalidWindowArtifactManifest;
    try artifactShapeEquals(try artifactField(entry, "shape"), shape);
    const expected_bytes = try artifactU32(try artifactField(entry, "bytes"));
    const bytes = try directory.readFileAlloc(io, file, allocator, .limited(1 << 30));
    defer allocator.free(bytes);
    if (bytes.len != expected_bytes) return error.InvalidWindowArtifactManifest;
    try artifactHashMatches(bytes, try artifactString(try artifactField(entry, "sha256")));
    return zml.Buffer.fromBytes(
        io,
        platform,
        zml.Shape.init(shape, if (std.mem.eql(u8, dtype, "float16")) .f16 else if (std.mem.eql(u8, dtype, "float32")) .f32 else if (std.mem.eql(u8, dtype, "int32")) .i32 else .u8),
        .replicated,
        bytes,
    );
}

/// Validate the recovery portion of a disposable execution fixture before
/// any ZML candidate is compiled. Raw rank8 buffers without the independent
/// audit, base/factor hashes, and certified arithmetic signature are rejected
/// so fixture tuning cannot bypass the unified artifact contract.
pub fn validateFixtureRecoveryManifest(root: std.json.ObjectMap) !void {
    const files = try artifactObject(root.get("files") orelse return error.InvalidFixtureRecovery);
    var factor_bytes: u32 = 0;
    for ([_][]const u8{ "rank8_a", "rank8_b" }) |name| {
        const entry = try artifactObject(files.get(name) orelse return error.InvalidFixtureRecovery);
        factor_bytes += try artifactU32(try artifactField(entry, "bytes"));
    }
    if (factor_bytes == 0) return;
    const recovery = try artifactObject(root.get("recovery") orelse return error.InvalidFixtureRecovery);
    if (try artifactU32(try artifactField(recovery, "rank")) != 8 or
        !std.mem.eql(u8, try artifactString(try artifactField(recovery, "dtype")), "float16") or
        !std.mem.eql(u8, try artifactString(try artifactField(recovery, "input_domain")), "p32_transformed") or
        !try artifactBool(try artifactField(recovery, "validated")) or
        !try artifactBool(try artifactField(recovery, "audit_validated")))
        return error.InvalidFixtureRecovery;
    const acceptance = try artifactObject(try artifactField(recovery, "audit_acceptance"));
    if (!try artifactBool(try artifactField(acceptance, "accepted")))
        return error.InvalidFixtureRecovery;
    if (recovery.get("arithmetic_signature")) |signature| {
        if (!std.mem.eql(u8, try artifactString(signature), "reference_fp32_v1"))
            return error.InvalidFixtureRecovery;
    }
    _ = try artifactString(try artifactField(recovery, "base_hash"));
    _ = try artifactString(try artifactField(recovery, "factors_hash"));
}

fn emptyArtifactTensor(io: std.Io, platform: *const zml.Platform) !zml.Buffer {
    return zml.Buffer.fromBytes(io, platform, zml.Shape.init(.{0}, .f16), .replicated, &.{});
}

fn powerOfTwo(value: u32) bool {
    return value != 0 and (value & (value - 1)) == 0;
}

fn nativeWindowShapeSupported(k: u32, n: u32, input_hadamard: bool, output_hadamard: bool) bool {
    if (k < 2048 or k > 16384 or k % 256 != 0 or
        n < 256 or n > 17408 or n % 256 != 0)
        return false;
    return (!input_hadamard or powerOfTwo(k)) and
        (!output_hadamard or powerOfTwo(n));
}

/// Load and validate the unified Python window artifact before device upload.
/// The loader owns the resulting device buffers; callers must keep the Artifact
/// alive until every executable and prepared graph using its buffers is gone.
pub fn loadArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    path: []const u8,
    options: ArtifactOptions,
) !Artifact {
    var directory = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer directory.close(io);
    const manifest_bytes = try directory.readFileAlloc(io, "manifest.json", allocator, .limited(1 << 20));
    defer allocator.free(manifest_bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_bytes, .{});
    defer parsed.deinit();
    const root = try artifactObject(parsed.value);
    if (!std.mem.eql(u8, try artifactString(try artifactField(root, "format")), "qvq_p32_window_artifact"))
        return error.UnsupportedWindowArtifactFormat;
    if (try artifactU32(try artifactField(root, "version")) != 1)
        return error.UnsupportedWindowArtifactVersion;
    const metadata = try artifactObject(try artifactField(root, "metadata"));
    const tensors = try artifactObject(try artifactField(root, "tensors"));
    try artifactValidateTensorNames(tensors);
    const expected_payload_hash = try artifactString(try artifactField(root, "payload_sha256"));
    const actual_payload_hash = try artifactBindingHex(metadata, tensors);
    if (!std.mem.eql(u8, &actual_payload_hash, expected_payload_hash))
        return error.WindowArtifactBindingMismatch;
    if (root.get("recovery")) |recovery_value| {
        if (recovery_value != .null) {
            const recovery_object = try artifactObject(recovery_value);
            const expected_base_hash = try artifactString(try artifactField(recovery_object, "base_hash"));
            const actual_base_hash = try artifactSemanticHash(allocator, io, &directory, metadata, tensors, false);
            if (!std.mem.eql(u8, &actual_base_hash, expected_base_hash))
                return error.WindowArtifactBaseHashMismatch;
            const expected_factors_hash = try artifactString(try artifactField(recovery_object, "factors_hash"));
            const actual_factors_hash = try artifactSemanticHash(allocator, io, &directory, metadata, tensors, true);
            if (!std.mem.eql(u8, &actual_factors_hash, expected_factors_hash))
                return error.WindowArtifactFactorsHashMismatch;
        }
    }
    const k = try artifactU32(try artifactField(metadata, "in_features"));
    const n = try artifactU32(try artifactField(metadata, "out_features"));
    const bits = try artifactField(metadata, "bits");
    const transition_bits: u32 = switch (bits) {
        .integer => |value| switch (value) {
            2 => 4,
            3 => 6,
            else => return error.UnsupportedWindowArtifactRate,
        },
        .float => |value| if (value == 2.5) 5 else if (value == 3.5) 7 else return error.UnsupportedWindowArtifactRate,
        else => return error.InvalidWindowArtifactManifest,
    };
    const input_hadamard = if (try artifactBool(try artifactField(metadata, "input_hadamard"))) @as(u32, 1) else 0;
    const output_hadamard = if (try artifactBool(try artifactField(metadata, "output_hadamard"))) @as(u32, 1) else 0;
    if (options.m == 0 or options.m > 8192 or
        !nativeWindowShapeSupported(k, n, input_hadamard != 0, output_hadamard != 0))
        return error.InvalidWindowArtifactShape;
    const tile_count = (k * n) / 256;
    if (k % 16 != 0 or n % 16 != 0 or tile_count == 0)
        return error.InvalidWindowArtifactShape;
    const bank_alt_entry = try artifactObject(tensors.get("bank_alt_id") orelse return error.InvalidWindowArtifactManifest);
    const bank_alt_file = try artifactString(try artifactField(bank_alt_entry, "file"));
    if (!std.mem.eql(u8, bank_alt_file, "bank_alt_id.bin") or
        !std.mem.eql(u8, try artifactString(try artifactField(bank_alt_entry, "dtype")), "uint8"))
        return error.InvalidWindowArtifactManifest;
    try artifactShapeEquals(try artifactField(bank_alt_entry, "shape"), &.{1});
    const bank_alt_bytes = try directory.readFileAlloc(io, bank_alt_file, allocator, .limited(16));
    defer allocator.free(bank_alt_bytes);
    if (bank_alt_bytes.len != 1 or try artifactU32(try artifactField(bank_alt_entry, "bytes")) != 1)
        return error.InvalidWindowArtifactShape;
    try artifactHashMatches(bank_alt_bytes, try artifactString(try artifactField(bank_alt_entry, "sha256")));
    const config = Config{
        .m = options.m,
        .k = k,
        .n = n,
        .transition_bits = transition_bits,
        .bank_alt_id = bank_alt_bytes[0],
        .algorithm = options.algorithm,
        .block_m = options.block_m,
        .block_n = options.block_n,
        .block_k = options.block_k,
        .warp_groups = options.warp_groups,
        .pipeline_stages = options.pipeline_stages,
        .split_k = options.split_k,
        .min_m = options.min_m,
        .max_m = options.max_m,
        .input_hadamard = input_hadamard,
        .output_hadamard = output_hadamard,
        .rank8_enabled = if (options.rank8_enabled) 1 else 0,
    };
    var payload = ArtifactPayload{
        .window = undefined,
        .banks = undefined,
        .levels = undefined,
        .su = undefined,
        .sv = undefined,
        .bias = undefined,
        .rank8_a = undefined,
        .rank8_b = undefined,
    };
    var initialized: [8]bool = @splat(false);
    errdefer {
        if (initialized[0]) payload.window.deinit();
        if (initialized[1]) payload.banks.deinit();
        if (initialized[2]) payload.levels.deinit();
        if (initialized[3]) payload.su.deinit();
        if (initialized[4]) payload.sv.deinit();
        if (initialized[5]) payload.bias.deinit();
        if (initialized[6]) payload.rank8_a.deinit();
        if (initialized[7]) payload.rank8_b.deinit();
    }
    payload.window = try artifactTensor(allocator, io, &directory, tensors, "window_words", "int32", &.{ tile_count, 4 * transition_bits }, platform);
    initialized[0] = true;
    payload.banks = try artifactTensor(allocator, io, &directory, tensors, "bank_ids", "uint8", &.{tile_count}, platform);
    initialized[1] = true;
    payload.levels = try artifactTensor(allocator, io, &directory, tensors, "levels", "float16", &.{256}, platform);
    initialized[2] = true;
    payload.su = try artifactTensor(allocator, io, &directory, tensors, "SU", "float32", &.{k}, platform);
    initialized[3] = true;
    payload.sv = try artifactTensor(allocator, io, &directory, tensors, "SV", "float32", &.{n}, platform);
    initialized[4] = true;
    if (tensors.get("bias")) |bias_value| {
        const bias_entry = try artifactObject(bias_value);
        const bias_dtype = try artifactString(try artifactField(bias_entry, "dtype"));
        if (std.mem.eql(u8, bias_dtype, "float16")) {
            payload.bias = try artifactTensor(allocator, io, &directory, tensors, "bias", "float16", &.{n}, platform);
        } else if (std.mem.eql(u8, bias_dtype, "float32")) {
            payload.bias = try artifactTensor(allocator, io, &directory, tensors, "bias", "float32", &.{n}, platform);
        } else {
            return error.InvalidWindowArtifactManifest;
        }
    } else {
        payload.bias = try emptyArtifactTensor(io, platform);
    }
    initialized[5] = true;
    const recovery = root.get("recovery") orelse .null;
    if (recovery != .null) {
        const recovery_object = try artifactObject(recovery);
        // The native ABI currently implements the certified reference
        // arithmetic.  Older manifests may omit this advisory field, but a
        // producer that supplies an alternative signature must not silently
        // route it through the reference graph.
        if (recovery_object.get("arithmetic_signature")) |signature_value| {
            if (!std.mem.eql(u8, try artifactString(signature_value), "reference_fp32_v1"))
                return error.InvalidWindowArtifactRecovery;
        }
        if (try artifactU32(try artifactField(recovery_object, "rank")) != 8)
            return error.UnsupportedWindowArtifactRank;
        if (!std.mem.eql(u8, try artifactString(try artifactField(recovery_object, "dtype")), "float16") or
            !std.mem.eql(u8, try artifactString(try artifactField(recovery_object, "input_domain")), "p32_transformed") or
            !try artifactBool(try artifactField(recovery_object, "validated")))
            return error.InvalidWindowArtifactRecovery;
        // Fitting validation covers only train/held-out selection.  Native
        // deployment requires the independent confirmation gate as well.
        if (!try artifactBool(try artifactField(recovery_object, "audit_validated")))
            return error.InvalidWindowArtifactRecovery;
        const audit_acceptance = try artifactObject(try artifactField(recovery_object, "audit_acceptance"));
        if (!try artifactBool(try artifactField(audit_acceptance, "accepted")))
            return error.InvalidWindowArtifactRecovery;
        payload.rank8_a = try artifactTensor(allocator, io, &directory, tensors, "rank8_A", "float16", &.{ k, 8 }, platform);
        initialized[6] = true;
        payload.rank8_b = try artifactTensor(allocator, io, &directory, tensors, "rank8_B", "float16", &.{ 8, n }, platform);
        initialized[7] = true;
    } else {
        if (options.rank8_enabled or tensors.get("rank8_A") != null or tensors.get("rank8_B") != null)
            return error.InvalidWindowArtifactRecovery;
        payload.rank8_a = try emptyArtifactTensor(io, platform);
        initialized[6] = true;
        payload.rank8_b = try emptyArtifactTensor(io, platform);
        initialized[7] = true;
    }
    return .{ .config = config, .payload = payload };
}

/// Native Hopper configurations that ZML may compile and benchmark for one
/// shape. The list intentionally retains every supported BM/BN choice and
/// each producer placement: the winning geometry is shape-, device-, and
/// correction-state dependent.
// Seven Hopper geometries x (four producer modes with two kernel variants,
// plus project-output with fused and fully-fused variants).  Composite or
// N>16384 outputs omit the last two candidates, but retain this fixed bound
// for stack allocation and ABI-stable reports.
pub const max_candidate_count: usize = 70;

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
    /// The producer must set this from its arithmetic certification record;
    /// reference is the safe default for the existing native ABI.
    arithmetic_signature: ArithmeticSignature = .reference_fp32_v1,
    /// Matched complete-operator correction-off/on timing.  This is optional
    /// for report-only tuning; an explicit recovery budget requires it.
    recovery_pair: ?RecoveryPairMeasurement = null,
};

pub const TuningResult = struct {
    config: Config,
    median_ns: u64,
    candidate_index: usize,
};

pub const BenchmarkOptions = struct {
    warmup_calls: usize = 2,
    iterations: usize = 16,
};

/// Matched correction-off/on timing for one otherwise identical executable
/// configuration.  The two executables must have been compiled from the same
/// shape, device, artifact and launch geometry; only the validated rank8
/// state may differ.  ZML keeps this result as tuning telemetry rather than
/// allowing latency to override the quantizer's quality decision.
pub const RecoveryPairMeasurement = struct {
    off_median_ns: u64,
    on_median_ns: u64,
    overhead_ns: i64,
    overhead_percent: f64,

    pub fn meetsTarget(self: @This(), maximum_percent: f64) bool {
        return self.off_median_ns > 0 and self.on_median_ns > 0 and
            std.math.isFinite(self.overhead_percent) and
            std.math.isFinite(maximum_percent) and maximum_percent >= 0 and
            self.overhead_percent <= maximum_percent;
    }
};

/// Build a correction-off/on result from already measured complete-executable
/// medians.  Keeping this pure makes cache/report code testable without a CUDA
/// runtime and rejects the invalid zero-off baseline explicitly.
pub fn recoveryPairFromMedians(off_median_ns: u64, on_median_ns: u64) !RecoveryPairMeasurement {
    if (off_median_ns == 0 or on_median_ns == 0) return error.InvalidRecoveryBaseline;
    const off = @as(f64, @floatFromInt(off_median_ns));
    const on = @as(f64, @floatFromInt(on_median_ns));
    return .{
        .off_median_ns = off_median_ns,
        .on_median_ns = on_median_ns,
        .overhead_ns = @as(i64, @intCast(on_median_ns)) - @as(i64, @intCast(off_median_ns)),
        .overhead_percent = (on / off - 1.0) * 100.0,
    };
}

/// Measure the same complete ZML executable twice, once with correction off
/// and once with validated rank8 correction on.  Compilation and graph
/// capture are intentionally outside this helper; both executables must be
/// prepared before calling it, and this function must never run from a
/// captured custom-call handler.
pub fn benchmarkRecoveryPair(
    allocator: std.mem.Allocator,
    io: std.Io,
    off_executable: *const zml.Exe,
    off_arguments: zml.Exe.Arguments,
    off_results: *zml.Exe.Results,
    on_executable: *const zml.Exe,
    on_arguments: zml.Exe.Arguments,
    on_results: *zml.Exe.Results,
    options: BenchmarkOptions,
) !RecoveryPairMeasurement {
    const off_median = try benchmarkExecutable(
        allocator,
        io,
        off_executable,
        off_arguments,
        off_results,
        options,
    );
    const on_median = try benchmarkExecutable(
        allocator,
        io,
        on_executable,
        on_arguments,
        on_results,
        options,
    );
    return recoveryPairFromMedians(off_median, on_median);
}

/// Measure one already-compiled candidate before serving-graph capture.
/// PJRT result readiness is included so callers compare complete executable
/// latency rather than an unrepresentative launch-only timestamp. The helper
/// allocates its samples and waits on results only in this tuning phase; it is
/// invalid to invoke it from a custom-call handler or captured graph.
pub fn benchmarkExecutable(
    allocator: std.mem.Allocator,
    io: std.Io,
    executable: *const zml.Exe,
    arguments: zml.Exe.Arguments,
    results: *zml.Exe.Results,
    options: BenchmarkOptions,
) !u64 {
    if (options.iterations == 0) return error.InvalidBenchmarkIterations;
    for (0..options.warmup_calls) |_| {
        executable.call(arguments, results);
        var output = results.get(zml.Buffer);
        try output.await(io);
        output.deinit();
    }
    const samples = try allocator.alloc(u64, options.iterations);
    defer allocator.free(samples);
    for (samples) |*sample| {
        const start: std.Io.Timestamp = .now(io, .awake);
        executable.call(arguments, results);
        var output = results.get(zml.Buffer);
        try output.await(io);
        sample.* = @intCast(start.untilNow(io, .awake).toNanoseconds());
        output.deinit();
    }
    std.sort.heap(u64, samples, {}, std.sort.asc(u64));
    return samples[samples.len / 2];
}

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
    return selectFastestWithPolicy(candidates, measurements, null, .balanced);
}

/// Select the fastest locally correct candidate, optionally requiring a
/// measured correction-off/on pair at or below `maximum_percent`.  The
/// default selector remains report-only for compatibility, while promotion
/// callers use this gate after benchmarking every candidate outside capture.
/// A missing pair is rejected when a budget is supplied; latency alone never
/// makes an unmeasured recovery candidate eligible.
pub fn selectFastestWithRecoveryGate(
    candidates: []const Config,
    measurements: []const CandidateMeasurement,
    maximum_percent: ?f64,
) !TuningResult {
    return selectFastestWithPolicy(candidates, measurements, maximum_percent, .balanced);
}

/// Select the fastest candidate under an explicit arithmetic and recovery
/// policy.  All measurements must already have been collected outside graph
/// capture.  `quality` admits only the reference FP32 signature; `balanced`
/// admits reference plus certified Tensor Core arithmetic; `fast` admits any
/// finite locally-correct candidate and is intended for correction-off graphs.
pub fn selectFastestWithPolicy(
    candidates: []const Config,
    measurements: []const CandidateMeasurement,
    maximum_percent: ?f64,
    quality_mode: QualityMode,
) !TuningResult {
    if (candidates.len == 0) return error.NoCandidates;
    if (candidates.len != measurements.len) return error.MeasurementCountMismatch;
    if (maximum_percent) |limit| {
        if (!std.math.isFinite(limit) or limit < 0) return error.InvalidRecoveryBudget;
    }
    var selected: ?TuningResult = null;
    for (candidates, measurements, 0..) |candidate, measurement, index| {
        // A correction-off graph does not execute the rank8 producer, so its
        // arithmetic signature cannot change the output contract.  Keep every
        // locally-correct launch candidate eligible in that state, including
        // unverified producer variants.  Once rank8 is enabled, the requested
        // quality policy remains authoritative.
        const arithmetic_allowed = candidate.rank8_enabled == 0 or
            arithmeticAllowed(quality_mode, measurement.arithmetic_signature);
        if (!measurement.accepted or measurement.median_ns == 0 or
            !arithmetic_allowed or
            !std.math.isFinite(measurement.mean_absolute_error) or
            !std.math.isFinite(measurement.max_absolute_error) or
            measurement.mean_absolute_error > 2e-3 or
            measurement.max_absolute_error > 3.0 / 64.0) continue;
        if (maximum_percent) |limit| {
            // Recovery overhead is undefined for an off graph.  An explicit
            // budget constrains enabled correction only; off-state tuning must
            // still be able to select the fastest valid geometry.
            if (candidate.rank8_enabled != 0) {
                const pair = measurement.recovery_pair orelse continue;
                if (!pair.meetsTarget(limit)) continue;
            }
        }
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
    // Candidate enumeration is a shape-specific preparation operation.  Bind
    // every returned policy to the exact M that was measured so a selected
    // BM/BN configuration cannot be reused accidentally for another request
    // shape through its broader caller-supplied range.  The Python tuner
    // applies the same min_m=max_m contract before timing.
    if (!configValid(base)) return 0;
    var shape_base = base;
    shape_base.min_m = base.m;
    shape_base.max_m = base.m;
    // The native fused epilogue supports power-of-two output Hadamard. The
    // project-output kernel additionally requires N <= 16384; larger
    // transform-free outputs retain fused epilogue candidates only.
    const output_hadamard_fusable = base.output_hadamard == 0 or
        (base.n >= 16 and (base.n & (base.n - 1)) == 0);
    const project_output_fusable = output_hadamard_fusable and base.n <= 16384;
    var policy_count: usize = 4;
    if (output_hadamard_fusable) {
        policy_count = 8;
        if (project_output_fusable) policy_count += 2;
    }
    // Keep the projection dimension present for correction-off candidates as
    // well. This preserves identical indices between the off and on sweeps,
    // allowing each concurrent producer candidate to receive a matched
    // correction-off baseline without making the off graph touch A/B.
    const required_count = 7 * policy_count;
    if (output.len < required_count) return 0;
    var count: usize = 0;
    // Transform-free outputs can select the native fused rank8 epilogue. Keep
    // both implementation dimensions paired across correction-off and
    // correction-on sweeps so recovery overhead is measured against the same
    // candidate index. The off executable ignores both policies while
    // retaining the geometry, which gives the verifier a matched baseline.
    for ([_]u32{ 0, 1, 2, 3 }) |recovery_projection| {
        const kernel_count: usize = if (output_hadamard_fusable) 2 else 1;
        for (0..kernel_count) |kernel_index| {
            const recovery_kernel: u32 = if (kernel_count == 2) @intCast(kernel_index) else 0;
            var m16 = shape_base;
            m16.algorithm = 1;
            m16.block_m = 0;
            m16.block_n = 0;
            m16.warp_groups = 0;
            m16.recovery_kernel = recovery_kernel;
            m16.recovery_projection = recovery_projection;
            output[count] = m16;
            count += 1;
            for ([_]u32{ 32, 64, 128 }) |block_m| {
                for ([_]u32{ 64, 128 }) |block_n| {
                    var candidate = shape_base;
                    candidate.algorithm = 2;
                    candidate.block_m = block_m;
                    candidate.block_n = block_n;
                    // Zero lets the native launcher select its normal warp-group
                    // policy; callers may override it when their tuner supports it.
                    candidate.warp_groups = 0;
                    candidate.recovery_kernel = recovery_kernel;
                    candidate.recovery_projection = recovery_projection;
                    output[count] = candidate;
                    count += 1;
                }
            }
        }
    }
    if (project_output_fusable) {
        for ([_]u32{ 1, 2 }) |recovery_kernel| {
            var m16 = shape_base;
            m16.algorithm = 1;
            m16.block_m = 0;
            m16.block_n = 0;
            m16.warp_groups = 0;
            m16.recovery_kernel = recovery_kernel;
            m16.recovery_projection = 4;
            output[count] = m16;
            count += 1;
            for ([_]u32{ 32, 64, 128 }) |block_m| {
                for ([_]u32{ 64, 128 }) |block_n| {
                    var candidate = shape_base;
                    candidate.algorithm = 2;
                    candidate.block_m = block_m;
                    candidate.block_n = block_n;
                    candidate.warp_groups = 0;
                    candidate.recovery_kernel = recovery_kernel;
                    candidate.recovery_projection = 4;
                    output[count] = candidate;
                    count += 1;
                }
            }
        }
    }
    return count;
}

/// Enumerate the same executable candidate set with deterministic
/// shape-aware priority. Every supported BM/BN choice remains present so the
/// tuner can discover wins that do not match this heuristic; compatible tiles
/// are simply measured first. This is preparation-time ordering only.
pub fn enumerateCandidatesForShape(base: Config, output: []Config) usize {
    const count = enumerateCandidates(base, output);
    if (count < 2) return count;
    var index: usize = 1;
    while (index < count) : (index += 1) {
        const value = output[index];
        const value_score = candidateShapeScoreForShape(base, value);
        var insert = index;
        while (insert > 0 and candidateShapeScoreForShape(base, output[insert - 1]) > value_score) {
            output[insert] = output[insert - 1];
            insert -= 1;
        }
        output[insert] = value;
    }
    return count;
}

/// Return the deterministic shape-priority score used by
/// `enumerateCandidatesForShape`. External ZML autotuners can record this
/// score alongside measured latency without duplicating the compatibility
/// policy. The score only orders candidates; it never removes one.
pub fn candidateShapeScoreForShape(base: Config, candidate: Config) i32 {
    // M16 is the safe small-M fallback. For larger batches direct tiles are
    // measured first, while all candidates remain in the returned set.
    var score: i32 = if (candidate.algorithm == 1)
        (if (base.m < 512) 0 else 200)
    else
        0;
    if (candidate.block_m != 0) {
        if (base.m < candidate.block_m) score += 50;
        if (base.m % candidate.block_m != 0) score += 100;
    }
    if (candidate.block_n != 0 and base.n % candidate.block_n != 0) score += 10;
    return score;
}

// Config is explicit compiler data: ZML may enumerate supported BM/BN/M
// candidates before lowering. rank8_enabled is resolved by quantizer metadata
// and requested quality mode, not chosen merely by latency.
pub fn linear(input: Input, config: Config) zml.Tensor {
    std.debug.assert(configValid(config));
    std.debug.assert(input.x.rank() == 2);
    std.debug.assert(input.x.dim(0) == config.m and input.x.dim(1) == config.k);
    inline for (@typeInfo(Input).@"struct".fields) |field| {
        const expected: zml.DataType = if (comptime std.mem.eql(u8, field.name, "window"))
            .i32
        else if (comptime std.mem.eql(u8, field.name, "banks"))
            .u8
        else
            .f16;
        if (comptime std.mem.eql(u8, field.name, "su") or
            std.mem.eql(u8, field.name, "sv") or
            std.mem.eql(u8, field.name, "bias"))
        {
            std.debug.assert(@field(input, field.name).dtype() == .f16 or
                @field(input, field.name).dtype() == .f32);
        } else {
            std.debug.assert(@field(input, field.name).dtype() == expected);
        }
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
    if (!configValid(config))
        return zml.pjrt.ffi.Error.create(frame.api, .invalid_argument, "invalid window configuration");
    const output = zml.pjrtx.CustomCallBuffer.fromPjrt(outputs[0]);
    buffers[9] = .{ .data = output.ptr, .bytes = output.shape.byteSize() };
    const device_ordinal: usize = @intCast(frame.ctx.getDeviceOrdinal(frame.api) catch {
        return zml.pjrt.ffi.Error.create(
            frame.api,
            .failed_precondition,
            "unable to identify PJRT device for native window graph",
        );
    });
    const stream: ?*anyopaque = @ptrCast(frame.api.stream(frame.ctx));
    var key: GraphKey = .{
        .buffers = undefined,
        .buffer_bytes = undefined,
        .buffer_types = undefined,
        .device_ordinal = device_ordinal,
        .stream = pointerValue(stream),
        .config = config,
    };
    for (inputs, 0..) |value, i| {
        const shape = zml.pjrtx.CustomCallBuffer.fromPjrt(value).shape;
        key.buffers[i] = pointerValue(buffers[i].data);
        key.buffer_bytes[i] = buffers[i].bytes;
        key.buffer_types[i] = @intFromEnum(shape.dtype());
    }
    key.buffers[9] = pointerValue(buffers[9].data);
    key.buffer_bytes[9] = buffers[9].bytes;
    key.buffer_types[9] = @intFromEnum(zml.pjrtx.CustomCallBuffer.fromPjrt(outputs[0]).shape.dtype());

    // A first eager call prepares the retained native graph. Once the backend
    // starts capture, a missing key is rejected by graph_create before any
    // allocation or event creation, so callers must warm each executable and
    // buffer set before capturing it.
    const handles = if (graph_handles) |*value| value else return zml.pjrt.ffi.Error.create(frame.api, .failed_precondition, "native window graph registry is not initialized");
    var handle: ?*anyopaque = null;
    var duplicate_handle: ?*anyopaque = null;
    var evicted_entry: ?*GraphEntry = null;
    graph_mutex.lock();
    graph_use_counter +%= 1;
    if (handles.getPtr(key)) |entry_ptr| {
        entry_ptr.*.last_used = graph_use_counter;
        handle = entry_ptr.*.handle;
    } else {
        // Do not hold the registry mutex while native graph creation may
        // allocate, synchronize, or compile. Another request may create the
        // same key concurrently; the duplicate is discarded after insertion.
        graph_mutex.unlock();
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
        const created_handle = handle.?;
        const new_entry = std.heap.c_allocator.create(GraphEntry) catch {
            destroyNativeGraph(created_handle);
            return zml.pjrt.ffi.Error.create(frame.api, .resource_exhausted, "unable to retain native window graph handle");
        };
        new_entry.* = .{ .handle = created_handle, .last_used = 0 };
        graph_mutex.lock();
        graph_use_counter +%= 1;
        if (handles.getPtr(key)) |existing_ptr| {
            existing_ptr.*.last_used = graph_use_counter;
            handle = existing_ptr.*.handle;
            duplicate_handle = created_handle;
            std.heap.c_allocator.destroy(new_entry);
        } else {
            new_entry.last_used = graph_use_counter;
            if (handles.count() >= max_graph_handles) {
                var oldest_key: ?GraphKey = null;
                var oldest_entry: ?*GraphEntry = null;
                var oldest_use: u64 = std.math.maxInt(u64);
                var iterator = handles.iterator();
                while (iterator.next()) |entry| {
                    // Never wait for a live replay while holding the global
                    // registry lock.  An entry is evictable only when its
                    // per-key lock can be acquired immediately; otherwise
                    // keep searching for an idle entry.
                    if (entry.value_ptr.*.last_used < oldest_use and
                        entry.value_ptr.*.run_lock.tryLock())
                    {
                        if (oldest_entry) |previous| previous.run_lock.unlock();
                        oldest_use = entry.value_ptr.*.last_used;
                        oldest_key = entry.key_ptr.*;
                        oldest_entry = entry.value_ptr.*;
                    }
                }
                if (oldest_key) |evicted| {
                    if (handles.fetchRemove(evicted)) |removed| {
                        // The selected entry is already locked, so native
                        // destruction can be deferred until after the map
                        // lock is released without blocking unrelated keys.
                        evicted_entry = removed.value;
                    } else if (oldest_entry) |selected| {
                        selected.run_lock.unlock();
                    }
                } else {
                    graph_mutex.unlock();
                    std.heap.c_allocator.destroy(new_entry);
                    destroyNativeGraph(created_handle);
                    return zml.pjrt.ffi.Error.create(
                        frame.api,
                        .resource_exhausted,
                        "all native window graph handles are busy",
                    );
                }
            }
            handles.put(key, new_entry) catch {
                graph_mutex.unlock();
                std.heap.c_allocator.destroy(new_entry);
                destroyNativeGraph(created_handle);
                if (evicted_entry) |old| {
                    destroyNativeGraph(old.handle);
                    old.run_lock.unlock();
                    std.heap.c_allocator.destroy(old);
                }
                return zml.pjrt.ffi.Error.create(frame.api, .resource_exhausted, "unable to retain native window graph handle");
            };
            handle = created_handle;
        }
    }
    const run_entry = handles.getPtr(key) orelse {
        graph_mutex.unlock();
        return zml.pjrt.ffi.Error.create(frame.api, .internal, "native window graph registry lost its inserted handle");
    };
    // Acquire the per-key lock while the registry lock is held so LRU eviction
    // cannot remove this entry between lookup and replay. The registry lock is
    // released before native work begins, so unrelated keys remain concurrent.
    run_entry.*.run_lock.lock();
    graph_mutex.unlock();
    defer run_entry.*.run_lock.unlock();
    // Native graph teardown can synchronize its owning stream and must never
    // run while the global registry lock is held. The current entry remains
    // protected by run_lock for the duration of replay.
    if (duplicate_handle) |duplicate| destroyNativeGraph(duplicate);
    if (evicted_entry) |old| {
        destroyNativeGraph(old.handle);
        old.run_lock.unlock();
        std.heap.c_allocator.destroy(old);
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
    try std.testing.expectEqual(@as(usize, 84), @sizeOf(Config));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(NativeBuffer));
    var candidates: [max_candidate_count]Config = undefined;
    const count = enumerateCandidates(.{ .m = 33, .k = 2048, .n = 2048, .transition_bits = 4, .bank_alt_id = 2, .algorithm = 2, .output_hadamard = 1, .rank8_enabled = 1 }, &candidates);
    try std.testing.expectEqual(@as(usize, 70), count);
    var hadamard_only: [max_candidate_count]Config = undefined;
    try std.testing.expectEqual(@as(usize, 70), enumerateCandidates(.{
        .m = 33,
        .k = 2048,
        .n = 2048,
        .transition_bits = 4,
        .bank_alt_id = 2,
        .algorithm = 2,
        .output_hadamard = 1,
    }, &hadamard_only));
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
    for (&measurements) |*measurement| {
        measurement.accepted = true;
        measurement.max_absolute_error = 0;
        measurement.median_ns = 100;
    }
    measurements[3].median_ns = 20;
    try std.testing.expectError(
        error.NoViableCandidate,
        selectFastestWithRecoveryGate(candidates[0..count], &measurements, 5),
    );
    try std.testing.expectError(
        error.InvalidBenchmarkIterations,
        benchmarkExecutable(std.testing.allocator, std.testing.io, undefined, undefined, undefined, .{ .iterations = 0 }),
    );
    const recovery = try recoveryPairFromMedians(100, 103);
    try std.testing.expectEqual(@as(i64, 3), recovery.overhead_ns);
    try std.testing.expectApproxEqAbs(@as(f64, 3), recovery.overhead_percent, 1e-12);
    try std.testing.expect(recovery.meetsTarget(5));
    try std.testing.expect(!recovery.meetsTarget(2));
    try std.testing.expect(!(RecoveryPairMeasurement{
        .off_median_ns = 0,
        .on_median_ns = 1,
        .overhead_ns = 1,
        .overhead_percent = 0,
    }).meetsTarget(5));
    measurements[0].recovery_pair = recovery;
    measurements[1].recovery_pair = try recoveryPairFromMedians(100, 108);
    measurements[2].recovery_pair = try recoveryPairFromMedians(100, 104);
    measurements[3].recovery_pair = try recoveryPairFromMedians(100, 120);
    const gated = try selectFastestWithRecoveryGate(candidates[0..count], &measurements, 5);
    try std.testing.expectEqual(@as(usize, 0), gated.candidate_index);
    measurements[1].median_ns = 1;
    measurements[1].arithmetic_signature = .certified_tensor_core_v1;
    measurements[3].arithmetic_signature = .certified_tensor_core_v1;
    const balanced = try selectFastestWithPolicy(candidates[0..count], &measurements, null, .balanced);
    try std.testing.expectEqual(@as(usize, 1), balanced.candidate_index);
    const quality = try selectFastestWithPolicy(candidates[0..count], &measurements, null, .quality);
    try std.testing.expectEqual(@as(usize, 0), quality.candidate_index);
    measurements[1].arithmetic_signature = .unverified;
    // A paired recovery cost is diagnostic unless the caller supplies an
    // explicit budget.  Keep a valid improvement selectable even at 20%
    // overhead; the 3--6% scorecard target is not an implicit rejection gate.
    measurements[1].recovery_pair = try recoveryPairFromMedians(100, 120);
    const fast = try selectFastestWithPolicy(candidates[0..count], &measurements, null, .fast);
    try std.testing.expectEqual(@as(usize, 1), fast.candidate_index);
    const capped = try selectFastestWithRecoveryGate(candidates[0..count], &measurements, 5);
    try std.testing.expectEqual(@as(usize, 0), capped.candidate_index);
    try std.testing.expectError(
        error.InvalidRecoveryBudget,
        selectFastestWithRecoveryGate(candidates[0..count], &measurements, -1),
    );
    try std.testing.expectError(error.InvalidRecoveryBaseline, recoveryPairFromMedians(0, 1));

    // Correction-off tuning may inspect unverified producer rows because the
    // producer is unreachable in this graph.  A budget is likewise irrelevant
    // until rank8 is enabled; the fastest valid off candidate remains eligible
    // even without a paired on-state measurement.
    var off_candidates = candidates;
    off_candidates[1].rank8_enabled = 0;
    measurements[1].accepted = true;
    measurements[1].median_ns = 1;
    measurements[1].arithmetic_signature = .unverified;
    measurements[1].recovery_pair = null;
    const off_balanced = try selectFastestWithPolicy(
        off_candidates[0..count],
        &measurements,
        null,
        .balanced,
    );
    try std.testing.expectEqual(@as(usize, 1), off_balanced.candidate_index);
    const off_budgeted = try selectFastestWithRecoveryGate(
        off_candidates[0..count],
        &measurements,
        5,
    );
    try std.testing.expectEqual(@as(usize, 1), off_budgeted.candidate_index);

    // The same unverified row is still rejected when the graph actually
    // enables rank8 and asks for a balanced arithmetic contract.
    off_candidates[1].rank8_enabled = 1;
    try std.testing.expectEqual(
        @as(usize, 3),
        (try selectFastestWithPolicy(
            off_candidates[0..count],
            &measurements,
            null,
            .balanced,
        )).candidate_index,
    );
    std.testing.refAllDecls(@This());
}

test "window configuration validation rejects unsafe external policies" {
    const valid = Config{
        .m = 96,
        .k = 2048,
        .n = 2048,
        .transition_bits = 4,
        .bank_alt_id = 2,
        .algorithm = 2,
        .block_m = 64,
        .block_n = 64,
        .warp_groups = 1,
    };
    try std.testing.expect(configValid(valid));
    var wrong_range = valid;
    wrong_range.min_m = 97;
    try std.testing.expect(!configValid(wrong_range));
    var wrong_geometry = valid;
    wrong_geometry.block_n = 32;
    try std.testing.expect(!configValid(wrong_geometry));
    var wrong_recovery = valid;
    wrong_recovery.recovery_projection = 5;
    try std.testing.expect(!configValid(wrong_recovery));
    var invalid_fully_fused = valid;
    invalid_fully_fused.rank8_enabled = 1;
    invalid_fully_fused.recovery_kernel = 2;
    invalid_fully_fused.recovery_projection = 3;
    try std.testing.expect(!configValid(invalid_fully_fused));
    var valid_project_output = valid;
    valid_project_output.rank8_enabled = 1;
    valid_project_output.recovery_kernel = 2;
    valid_project_output.recovery_projection = 4;
    try std.testing.expect(configValid(valid_project_output));
    var wrong_algorithm = valid;
    wrong_algorithm.algorithm = 0;
    try std.testing.expect(!configValid(wrong_algorithm));
}

test "transform-free candidate enumeration exposes fused rank8 policy" {
    var candidates: [max_candidate_count]Config = undefined;
    const count = enumerateCandidates(.{
        .m = 33,
        .k = 2048,
        .n = 2048,
        .transition_bits = 4,
        .bank_alt_id = 2,
        .algorithm = 2,
        .output_hadamard = 0,
        .rank8_enabled = 1,
    }, &candidates);
    try std.testing.expectEqual(@as(usize, 70), count);
    try std.testing.expectEqual(@as(u32, 0), candidates[0].recovery_kernel);
    try std.testing.expectEqual(@as(u32, 1), candidates[7].recovery_kernel);
    try std.testing.expectEqual(@as(u32, 1), candidates[14].recovery_projection);
    try std.testing.expectEqual(@as(u32, 2), candidates[28].recovery_projection);
    try std.testing.expectEqual(candidates[0].block_m, candidates[7].block_m);
    try std.testing.expectEqual(candidates[6].block_n, candidates[13].block_n);
    try std.testing.expectEqual(candidates[0].block_m, candidates[14].block_m);
    try std.testing.expectEqual(candidates[6].block_n, candidates[20].block_n);
}

test "native window shape policy admits composite transform-free Qwen tiles" {
    try std.testing.expect(nativeWindowShapeSupported(5120, 17408, false, false));
    try std.testing.expect(!nativeWindowShapeSupported(5120, 17408, true, false));
    try std.testing.expect(!nativeWindowShapeSupported(5120, 17408, false, true));
    try std.testing.expect(nativeWindowShapeSupported(2048, 2048, true, true));
    try std.testing.expect(!nativeWindowShapeSupported(2048, 17920, false, false));
}

test "candidate enumeration omits project-output for unsupported widths" {
    var candidates: [max_candidate_count]Config = undefined;
    const count = enumerateCandidates(.{
        .m = 33,
        .k = 5120,
        .n = 17408,
        .transition_bits = 4,
        .bank_alt_id = 2,
        .algorithm = 2,
        .output_hadamard = 0,
        .rank8_enabled = 1,
    }, &candidates);
    try std.testing.expectEqual(@as(usize, 56), count);
    for (candidates[0..count]) |candidate| {
        try std.testing.expect(candidate.recovery_projection != 4);
    }
}

test "shape-aware candidate ordering keeps every geometry" {
    var candidates: [max_candidate_count]Config = undefined;
    const count = enumerateCandidatesForShape(.{
        .m = 96,
        .k = 2048,
        .n = 2048,
        .transition_bits = 4,
        .bank_alt_id = 2,
        .algorithm = 2,
        .output_hadamard = 1,
    }, &candidates);
    try std.testing.expectEqual(@as(usize, 70), count);
    try std.testing.expectEqual(@as(u32, 1), candidates[0].algorithm);
    try std.testing.expectEqual(@as(u32, 32), candidates[1].block_m);
    const base = candidates[0];
    try std.testing.expectEqual(@as(i32, 0), candidateShapeScoreForShape(base, candidates[0]));
    try std.testing.expect(candidateShapeScoreForShape(base, candidates[1]) >= 0);
    var saw_bm64 = false;
    var saw_bm128 = false;
    for (candidates[0..count]) |candidate| {
        saw_bm64 = saw_bm64 or candidate.block_m == 64;
        saw_bm128 = saw_bm128 or candidate.block_m == 128;
        try std.testing.expectEqual(@as(u32, 96), candidate.min_m);
        try std.testing.expectEqual(@as(u32, 96), candidate.max_m);
    }
    try std.testing.expect(saw_bm64 and saw_bm128);
}

test "candidate enumeration rejects an unbound M shape" {
    var candidates: [max_candidate_count]Config = undefined;
    try std.testing.expectEqual(@as(usize, 0), enumerateCandidates(.{
        .m = 0,
        .k = 2048,
        .n = 2048,
        .transition_bits = 4,
        .bank_alt_id = 2,
        .algorithm = 2,
    }, &candidates));
}

test "graph identity includes buffer layout and element dtype" {
    const left: GraphKey = .{
        .buffers = @splat(11),
        .buffer_bytes = @splat(32),
        .buffer_types = @splat(1),
        .device_ordinal = 0,
        .stream = 7,
        .config = std.mem.zeroes(Config),
    };
    var right = left;
    try std.testing.expect(GraphKeyContext.eql(.{}, left, right));
    right.buffer_bytes[0] = 64;
    try std.testing.expect(!GraphKeyContext.eql(.{}, left, right));
    right = left;
    right.buffer_types[0] = 2;
    try std.testing.expect(!GraphKeyContext.eql(.{}, left, right));
    right = left;
    right.device_ordinal = 1;
    try std.testing.expect(!GraphKeyContext.eql(.{}, left, right));
}

test "graph registry replay locks the retained entry" {
    var handles = GraphMap.init(std.testing.allocator);
    defer handles.deinit();
    const key: GraphKey = .{
        .buffers = @splat(19),
        .buffer_bytes = @splat(64),
        .buffer_types = @splat(1),
        .device_ordinal = 0,
        .stream = 3,
        .config = std.mem.zeroes(Config),
    };
    const entry = try std.testing.allocator.create(GraphEntry);
    entry.* = .{ .handle = null, .last_used = 1 };
    try handles.put(key, entry);
    const retained = handles.getPtr(key) orelse return error.TestUnexpectedResult;
    // `getPtr` returns the map's retained pointer value. Replay must lock
    // this object, rather than copying GraphEntry (and therefore its mutex).
    try std.testing.expectEqual(@intFromPtr(entry), @intFromPtr(retained.*));
    _ = handles.fetchRemove(key);
    std.testing.allocator.destroy(entry);
}

test "graph eviction probes busy entries without blocking" {
    var lock: GraphLock = .{};
    try std.testing.expect(lock.tryLock());
    // The eviction path must skip a live replay instead of waiting while the
    // global registry map is held.
    try std.testing.expect(!lock.tryLock());
    lock.unlock();
    try std.testing.expect(lock.tryLock());
    lock.unlock();
}
