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
const NativeFunction = *const fn (
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    NativeBuffer,
    *const Config,
    ?*anyopaque,
    [*]u8,
    u64,
) callconv(.c) c_int;
var native_function: ?NativeFunction = null;

pub const Runtime = struct {
    libraries: [3]std.DynLib,

    // Load existing QVQ CUDA, Hopper WGMMA, then the native window ABI library.
    // Keep this runtime alive until every executable using it has finished.
    pub fn init(paths: [3][]const u8, platform: *const zml.Platform) !Runtime {
        if (native_function != null) return error.AlreadyInitialized;
        var libraries: [3]std.DynLib = undefined;
        var loaded: usize = 0;
        errdefer for (libraries[0..loaded]) |*library| library.close();
        for (paths, 0..) |path, i| {
            libraries[i] = try std.DynLib.open(path);
            loaded += 1;
        }
        native_function = libraries[2].lookup(NativeFunction, "qvq_p32_window_linear") orelse
            return error.MissingNativeWindowSymbol;
        errdefer native_function = null;
        try platform.registerFfi(.{
            .name = "qvq_p32_window_linear",
            .handler = handler,
            // The initial reference ABI allocates ATen temporaries. It must
            // not advertise external command-buffer/capture compatibility.
            .traits = .{ .command_buffer_compatible = false },
        });
        return .{ .libraries = libraries };
    }

    pub fn deinit(self: *Runtime) void {
        native_function = null;
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
    const function = native_function orelse
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
    var message: [4096]u8 = @splat(0);
    const status = function(
        buffers[0],
        buffers[1],
        buffers[2],
        buffers[3],
        buffers[4],
        buffers[5],
        buffers[6],
        buffers[7],
        buffers[8],
        buffers[9],
        &config,
        @ptrCast(frame.api.stream(frame.ctx)),
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
    std.testing.refAllDecls(@This());
}
