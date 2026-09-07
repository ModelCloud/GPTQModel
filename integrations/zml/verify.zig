// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const zml = @import("zml");
const window = @import("qvq_window.zig");

fn run(input: window.Input, config: window.Config) zml.Tensor {
    return window.linear(input, config);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, struct { fixture: []const u8 });
    var directory = try std.Io.Dir.cwd().openDir(io, args.fixture, .{});
    defer directory.close(io);
    const metadata = try directory.readFileAlloc(io, "manifest.json", allocator, .limited(1 << 20));
    defer allocator.free(metadata);
    const manifest = try std.json.parseFromSlice(std.json.Value, allocator, metadata, .{});
    defer manifest.deinit();
    const parsed = try std.json.parseFromValue(window.Config, allocator, manifest.value.object.get("config").?, .{});
    defer parsed.deinit();
    var config = parsed.value;
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
    for (0..2) |enabled| {
        config.rank8_enabled = @intCast(enabled);
        var executable = try platform.compileFn(allocator, io, run, .{ tensors, config }, .{});
        defer executable.deinit();
        var arguments = try executable.args(allocator);
        defer arguments.deinit(allocator);
        var results = try executable.results(allocator);
        defer results.deinit(allocator);
        arguments.set(.{ buffers, config });
        // The first eager call prepares and retains the native graph. The
        // second call exercises ordinary replay through the same handle and
        // catches accidental per-call graph creation or pointer churn.
        for (0..2) |_| executable.call(arguments, &results);
        var output = results.get(zml.Buffer);
        defer output.deinit();
        const actual = try output.toSliceAlloc(allocator, io);
        defer actual.free(allocator);
        const expected = try directory.readFileAlloc(
            io,
            if (enabled == 1) "expected_on.bin" else "expected_off.bin",
            allocator,
            .limited(1 << 30),
        );
        defer allocator.free(expected);
        if (!std.mem.eql(u8, actual.bytes, expected)) return error.WindowOutputNotBitExact;
        std.log.info("ZML window rank8={d}: bit-exact {d}x{d} output", .{ enabled, config.m, config.n });
    }
}
