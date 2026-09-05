// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// Linux x86-64 ISA availability probe, not a performance or model-quality test.
// Build: gcc -O2 -mamx-tile -mamx-bf16 -mamx-int8 scripts/probe_qvq_cpu_amx.c -o /tmp/qvq-amx-probe
// Requests tile-state permission in this process; changes no host CPU allocation.

#include <asm/prctl.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

struct tile_config {
    uint8_t palette, start_row, reserved[14];
    uint16_t cols[8];
    uint8_t reserved2[16], rows[8], reserved3[8];
};

int main(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("amx-tile") || !__builtin_cpu_supports("amx-bf16") ||
        !__builtin_cpu_supports("amx-int8")) {
        puts("{\"amx_execution\":\"unsupported\"}");
        return 0;
    }
    unsigned long supported = 0, before = 0, after = 0;
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_SUPP, &supported) ||
        syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &before) ||
        syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, ARCH_XCOMP_TILEDATA) ||
        syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &after)) {
        perror("arch_prctl AMX permission");
        return 2;
    }
    _Alignas(64) struct tile_config config = {.palette = 1};
    _Alignas(64) uint16_t bf_a[512], bf_b[512];
    _Alignas(64) int8_t int_a[1024], int_b[1024];
    _Alignas(64) float bf_out[256];
    _Alignas(64) int32_t int_out[256];
    for (int i = 0; i < 3; ++i) { config.cols[i] = 64; config.rows[i] = 16; }
    for (int i = 0; i < 512; ++i) bf_a[i] = bf_b[i] = 0x3f80;
    for (int i = 0; i < 1024; ++i) int_a[i] = int_b[i] = 1;
    _tile_loadconfig(&config);
    _tile_zero(0);
    _tile_loadd(1, bf_a, 64);
    _tile_loadd(2, bf_b, 64);
    _tile_dpbf16ps(0, 1, 2);
    _tile_stored(0, bf_out, 64);
    _tile_zero(0);
    _tile_loadd(1, int_a, 64);
    _tile_loadd(2, int_b, 64);
    _tile_dpbssd(0, 1, 2);
    _tile_stored(0, int_out, 64);
    _tile_release();
    int bf_ok = 1, int_ok = 1;
    for (int i = 0; i < 256; ++i) {
        bf_ok &= bf_out[i] == 32.0f;
        int_ok &= int_out[i] == 64;
    }
    printf("{\"amx_bf16_executed_and_checked\":%s,\"amx_int8_executed_and_checked\":%s,"
           "\"xcomp_supported\":\"0x%lx\",\"xcomp_permission_before\":\"0x%lx\","
           "\"xcomp_permission_after\":\"0x%lx\"}\n",
           bf_ok ? "true" : "false", int_ok ? "true" : "false", supported, before, after);
    return !(bf_ok && int_ok);
}
