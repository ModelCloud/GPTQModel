// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

// patch_cubin.cpp - in-place patch an sm_120a cubin to enable hardware
// narrow-float formats that PTX does not expose (only reachable by editing SASS
// encoding bits).
//
//   E0M3 : FP4 sign-magnitude integer {0,+/-1..+/-7}   (tensor-core OMMA / cvt F2FP)
//   E3M4 : hidden FP8 format, bias 3 / 4-mantissa / +/-30  (tensor-core QMMA / cvt F2FP)
//
// Constants below were established on RTX PRO 6000 (sm_120a, CUDA 13.3) by
// bit-flipping + on-GPU numeric regression. Each instruction is 16 bytes,
// little-endian; bit i lives in byte i/8, bit i%8.
//
//   OMMA (block-scaled FP4)  opcode[11:0] = 0x47f
//         bit78 = A element format (0=E2M1, 1=E0M3), bit79 = B element format
//   QMMA (FP8/6/4)           opcode[11:0] = 0x47a (block-scaled) or 0x27a (plain)
//         A type field = {78,82,83}, B type field = {79,84,85}, 3-bit LE codes:
//         000=E4M3 001=E5M2 010=E3M4(hidden) ...  E3M4 is emitted as E5M2, so
//         E5M2->E3M4: clear bit78 set bit82 (A) / clear bit79 set bit84 (B)
//   F2FP (narrow-float cvt)  opcode[11:0] = 0x23e, format subfield = (b76,b85,b86,b87)
//         FP8: E4M3=(1,0,0,1) E5M2=(0,0,0,1) E3M4=(0,1,0,1)  E5M2->E3M4: b85=1
//         FP4: E2M1=(1,1,0,0) E0M3=(0,1,0,0)                 E2M1->E0M3: b76=0
//
// Only instructions whose riginal bits are the expected base format (E5M2 for 
// e3m4 modes, E2M1 for e0m3 modes) are modified; anything else is skipped.
// A cubin that is not sm_120a is rejected outright.
//
// Usage:
//   patch_cubin <mode> <cubin> [--dry-run] [--backup]
//   mode = mma_e3m4_a|b|ab  mma_e0m3_a|b|ab  cvt_e3m4  cvt_e0m3
//   --backup writes <cubin>.bak (off by default; overwrites an existing .bak)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int getbit(const std::vector<uint8_t> &d, size_t base, int bit) {
  return (d[base + bit / 8] >> (bit % 8)) & 1;
}
static void setbit(std::vector<uint8_t> &d, size_t base, int bit, int v) {
  uint8_t m = 1u << (bit % 8);
  if (v) d[base + bit / 8] |= m;
  else d[base + bit / 8] &= ~m;
}
static uint32_t opcode12(const std::vector<uint8_t> &d, size_t base) {
  return d[base] | ((uint32_t)(d[base + 1] & 0x0f) << 8);
}

enum Kind {
  K_MMA_E3M4,
  K_MMA_E0M3,
  K_CVT_E3M4,
  K_CVT_E0M3
};

struct Mode {
  const char *name;
  Kind kind;
  bool a, b;
};

static const Mode MODES[] = {
    {"mma_e3m4_a", K_MMA_E3M4, true, false},
    {"mma_e3m4_b", K_MMA_E3M4, false, true},
    {"mma_e3m4_ab", K_MMA_E3M4, true, true},
    {"mma_e0m3_a", K_MMA_E0M3, true, false},
    {"mma_e0m3_b", K_MMA_E0M3, false, true},
    {"mma_e0m3_ab", K_MMA_E0M3, true, true},
    {"cvt_e3m4", K_CVT_E3M4, true, true},
    {"cvt_e0m3", K_CVT_E0M3, true, true},
};

struct Sec {
  std::string name;
  uint64_t off, size;
};

static bool read_file(const char *p, std::vector<uint8_t> &out) {
  FILE *f = fopen(p, "rb");
  if (!f) return false;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
  long n = ftell(f);
  if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return false; }
  out.resize(n);
  bool ok = (fread(out.data(), 1, n, f) == (size_t)n);
  fclose(f);
  return ok;
}

template <class T>
static T rd(const std::vector<uint8_t> &d, size_t o) {
  T v;
  memcpy(&v, &d[o], sizeof(T));
  return v;
}

// Accept only EM_CUDA (0xBE) with SM field == 120. The cubin e_flags are
// identical for sm_120 / sm_120a / sm_120f (0x06007802), so the a/f suffix
// cannot be told apart at the ELF level; block-scaled MMA only builds for
// sm_120a anyway, so gating on the SM120 family is sufficient.
static int check_sm120a(const std::vector<uint8_t> &d, std::string &why) {
  if (d.size() < 64 || d[0] != 0x7f || d[1] != 'E' || d[2] != 'L' || d[3] != 'F') {
    why = "not an ELF/cubin";
    return 1;
  }
  if (d[4] != 2 || d[5] != 1) {
    why = "only 64-bit little-endian cubin is supported";
    return 1;
  }
  uint16_t emachine = rd<uint16_t>(d, 0x12);
  uint32_t eflags = rd<uint32_t>(d, 0x30);
  if (emachine != 0xBE) {
    char b[64];
    snprintf(b, 64, "e_machine=0x%x is not a CUDA cubin", emachine);
    why = b;
    return 1;
  }
  uint32_t sm = (eflags >> 8) & 0xff;
  if (sm != 120) {
    char b[80];
    snprintf(b, 80, "target arch SM=%u (e_flags=0x%08x) is not sm_120", sm, eflags);
    why = b;
    return 1;
  }
  return 0;
}

static void collect_exec_sections(const std::vector<uint8_t> &d, std::vector<Sec> &secs) {
  if (d.size() < 64) return;
  uint64_t e_shoff = rd<uint64_t>(d, 0x28);
  uint16_t e_shents = rd<uint16_t>(d, 0x3A);
  uint16_t e_shnum = rd<uint16_t>(d, 0x3C);
  uint16_t e_shstr = rd<uint16_t>(d, 0x3E);
  uint64_t shstr = e_shoff + (uint64_t)e_shstr * e_shents;
  if (shstr + 0x20 > d.size()) return;
  uint64_t stroff = rd<uint64_t>(d, shstr + 0x18);
  for (uint16_t i = 0; i < e_shnum; i++) {
    uint64_t sh = e_shoff + (uint64_t)i * e_shents;
    if (sh + 0x28 > d.size()) break;
    uint32_t sh_name = rd<uint32_t>(d, sh + 0x00);
    uint32_t sh_type = rd<uint32_t>(d, sh + 0x04);
    uint64_t sh_flag = rd<uint64_t>(d, sh + 0x08);
    uint64_t sh_off = rd<uint64_t>(d, sh + 0x18);
    uint64_t sh_size = rd<uint64_t>(d, sh + 0x20);
    if (sh_type == 1 && (sh_flag & 0x4)) { // PROGBITS + EXECINSTR
      if (stroff + sh_name >= d.size()) continue; // guard section-name string
      const char *nm = (const char *)&d[stroff + sh_name];
      secs.push_back({std::string(nm), sh_off, sh_size});
    }
  }
}

// Returns 1 = patched, 0 = skipped (opcode matches but format doesn't), -1 = other opcode.
static int handle(std::vector<uint8_t> &d, size_t o, const Mode &m, bool dry, std::string &reason) {
  uint32_t op = opcode12(d, o);
  char buf[160];

  if (m.kind == K_MMA_E0M3) {
    if (op != 0x47f) return -1; // OMMA
    // E0M3 is only legal with scale_vec::4X (bit120). The 2X form (bit83) with
    // E0M3 is an illegal encoding that faults at launch, so refuse to patch it.
    if (getbit(d, o, 83)) {
      reason = "E0M3 OMMA requires scale_vec::4X but found 2X (bit83)";
      return -2; // fatal: unsupported instruction config
    }
    int did = 0;
    std::string r;
    auto side = [&](bool en, int bit, const char *tag) {
      if (!en) return;
      int cur = getbit(d, o, bit);
      if (cur == 0) {
        if (!dry) setbit(d, o, bit, 1);
        did = 1;
        r += std::string(tag) + "E2M1->E0M3 ";
      } else r += std::string(tag) + "already-E0M3(skip) ";
    };
    side(m.a, 78, "A:");
    side(m.b, 79, "B:");
    reason = r;
    return did ? 1 : 0;
  }
  if (m.kind == K_MMA_E3M4) {
    if (op != 0x47a && op != 0x27a) return -1; // QMMA
    int did = 0;
    std::string r;
    if (m.a) { // A field {78,82,83}==001(LE) is E5M2; E3M4==010 clears bit78, sets bit82
      int f78 = getbit(d, o, 78), f82 = getbit(d, o, 82), f83 = getbit(d, o, 83);
      if (f78 == 1 && f82 == 0 && f83 == 0) {
        if (!dry) { setbit(d, o, 78, 0); setbit(d, o, 82, 1); }
        did = 1;
        r += "A:E5M2->E3M4 ";
      } else {
        snprintf(buf, 160, "A:not-E5M2(%d%d%d,skip) ", f78, f82, f83);
        r += buf;
      }
    }
    if (m.b) { // B field {79,84,85}==001(LE) is E5M2; E3M4==010 clears bit79, sets bit84
      int f79 = getbit(d, o, 79), f84 = getbit(d, o, 84), f85 = getbit(d, o, 85);
      if (f79 == 1 && f84 == 0 && f85 == 0) {
        if (!dry) { setbit(d, o, 79, 0); setbit(d, o, 84, 1); }
        did = 1;
        r += "B:E5M2->E3M4 ";
      } else {
        snprintf(buf, 160, "B:not-E5M2(%d%d%d,skip) ", f79, f84, f85);
        r += buf;
      }
    }
    reason = r;
    return did ? 1 : 0;
  }
  // cvt (F2FP 0x23e), format subfield (b76,b85,b86,b87)
  if (op != 0x23e) return -1;
  int b76 = getbit(d, o, 76), b85 = getbit(d, o, 85), b86 = getbit(d, o, 86), b87 = getbit(d, o, 87);
  if (m.kind == K_CVT_E3M4) {
    if (b76 == 0 && b85 == 0 && b86 == 0 && b87 == 1) { // FP8 E5M2 -> E3M4
      if (!dry) {
        setbit(d, o, 85, 1);
      }
      reason = "E5M2->E3M4";
      return 1;
    }
    snprintf(buf, 160, "not-E5M2-down(b76,85,86,87=%d%d%d%d,skip)", b76, b85, b86, b87);
    reason = buf;
    return 0;
  } else {
    if (b76 == 1 && b85 == 1 && b86 == 0 && b87 == 0) { // FP4 E2M1 -> E0M3
      if (!dry) setbit(d, o, 76, 0);
      reason = "E2M1->E0M3";
      return 1;
    }
    snprintf(buf, 160, "not-E2M1-down(b76,85,86,87=%d%d%d%d,skip)", b76, b85, b86, b87);
    reason = buf;
    return 0;
  }
}

struct PatchStats {
  int rc = 0;      // 0 ok; 2 bad mode / read error; 3 not sm_120a; 4 I/O error
  int matched = 0; // instructions of the mode's opcode class
  int patched = 0; // instructions actually modified
  int skipped = 0; // matched opcode but wrong source format
  std::string message;
};

// Core routine: given a cubin path and a mode name, validate, scan, and (unless
// dry) write the patched file in place. When verbose, print a per-instruction
// report to stdout.
static PatchStats run_patch(const std::string &path, const std::string &mode_name,
                            bool dry, bool backup, bool verbose = false) {
  PatchStats st;
  const Mode *mode = nullptr;
  for (auto &M : MODES)
    if (mode_name == M.name) mode = &M;
  if (!mode) {
    st.rc = 2;
    st.message = "unknown mode: " + mode_name;
    return st;
  }

  std::vector<uint8_t> d;
  if (!read_file(path.c_str(), d)) {
    st.rc = 2;
    st.message = "cannot read: " + path;
    return st;
  }

  std::string why;
  if (check_sm120a(d, why)) {
    st.rc = 3;
    st.message = "refused: " + why;
    return st;
  }

  std::vector<Sec> secs;
  collect_exec_sections(d, secs);
  if (verbose)
    printf("mode=%s  file=%s  %s\n", mode->name, path.c_str(), dry ? "[dry-run]" : "[in-place]");
  for (auto &s : secs) {
    for (uint64_t p = s.off; p + 16 <= s.off + s.size; p += 16) {
      std::string reason;
      int r = handle(d, p, *mode, dry, reason);
      if (r == -1) continue;
      if (r == -2) { // fatal: unsupported instruction config, abort without writing
        st.rc = 5;
        st.message = reason;
        if (verbose)
          printf("  @0x%06llx [%s]  %s\n", (unsigned long long)p, s.name.c_str(), reason.c_str());
        return st;
      }
      st.matched++;
      if (r == 1) {
        st.patched++;
        if (verbose)
          printf("  @0x%06llx [%s]  %s%s\n", (unsigned long long)p, s.name.c_str(),
                 reason.c_str(), dry ? "  (would change)" : "  [changed]");
      } else {
        st.skipped++;
        if (verbose)
          printf("  @0x%06llx [%s]  %s\n", (unsigned long long)p, s.name.c_str(), reason.c_str());
      }
    }
  }
  if (verbose)
    printf("matched %d instruction(s) of this opcode class: %s %d, skipped %d.\n",
           st.matched, dry ? "to-change" : "changed", st.patched, st.skipped);

  if (dry) {
    if (verbose) printf("[dry-run] no file written.\n");
    st.message = "dry-run";
    return st;
  }
  if (st.patched == 0) {
    if (verbose) printf("no patchable instruction; file unchanged.\n");
    st.message = "no change";
    return st;
  }

  // Optional backup (only with --backup); always overwrites an existing .bak.
  if (backup) {
    std::string bak = path + ".bak";
    std::vector<uint8_t> orig;
    if (!read_file(path.c_str(), orig)) {
      st.rc = 4;
      st.message = "failed to re-read original; write aborted.";
      return st;
    }
    FILE *bw = fopen(bak.c_str(), "wb");
    if (!bw) {
      st.rc = 4;
      st.message = "cannot write backup " + bak;
      return st;
    }
    size_t wrote = fwrite(orig.data(), 1, orig.size(), bw);
    fclose(bw);
    if (wrote != orig.size()) {
      st.rc = 4;
      st.message = "failed to write backup " + bak;
      return st;
    }
    if (verbose) printf("backed up original -> %s\n", bak.c_str());
  }
  FILE *w = fopen(path.c_str(), "wb");
  if (!w) {
    st.rc = 4;
    st.message = "cannot write back " + path;
    return st;
  }
  size_t wrote = fwrite(d.data(), 1, d.size(), w);
  fclose(w);
  if (wrote != d.size()) {
    st.rc = 4;
    st.message = "failed to write patched cubin " + path;
    return st;
  }
  if (verbose) printf("patched in place %s (%d instruction(s) modified).\n", path.c_str(), st.patched);
  return st;
}

// C-ABI entry point for loading as a shared library from Python via ctypes:
//   g++ -O2 -std=c++17 -fPIC -shared patch_cubin.cpp -o libcubinpatch.so
// Returns the number of patched instructions (>=0), or -rc on error
// (2 bad mode/read, 3 not sm_120a, 4 I/O, 5 unsupported instruction config).
extern "C" int cubin_patch(const char *path, const char *mode, int dry, int backup) {
  PatchStats st = run_patch(path, mode, dry != 0, backup != 0, /*verbose=*/false);
  return st.rc ? -st.rc : st.patched;
}

// C-ABI entry point that patches a cubin already resident in memory (e.g. the
// bytes Triton is about to load). Length-preserving: only bit fields inside
// existing instructions are flipped, so the buffer is edited in place and its
// size never changes. `data` must point to `n` writable bytes.
// Returns the number of patched instructions (>=0), or -rc on error
// (2 bad mode, 3 not sm_120a, 5 unsupported instruction config). Nothing is
// written back when dry != 0.
extern "C" int cubin_patch_buffer(uint8_t *data, size_t n, const char *mode, int dry) {
  if (!data || n == 0 || !mode) return -2;
  const Mode *mode_p = nullptr;
  for (auto &M : MODES)
    if (strcmp(mode, M.name) == 0) mode_p = &M;
  if (!mode_p) return -2;

  std::vector<uint8_t> d(data, data + n);
  std::string why;
  if (check_sm120a(d, why)) return -3;

  std::vector<Sec> secs;
  collect_exec_sections(d, secs);
  int patched = 0;
  for (auto &s : secs) {
    for (uint64_t p = s.off; p + 16 <= s.off + s.size; p += 16) {
      std::string reason;
      if (handle(d, p, *mode_p, dry != 0, reason) == 1) patched++;
    }
  }
  if (dry == 0 && patched > 0) memcpy(data, d.data(), n);
  return patched;
}

int main(int argc, char **argv) {
  const char *mode_s = nullptr;
  const char *path = nullptr;
  bool dry = false, backup = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--dry-run") dry = true;
    else if (a == "--backup") backup = true;
    else if (!mode_s) mode_s = argv[i];
    else if (!path) path = argv[i];
    else {
      fprintf(stderr, "extra argument: %s\n", argv[i]);
      return 2;
    }
  }
  if (!mode_s || !path) {
    fprintf(stderr,
            "usage: %s <mode> <cubin> [--dry-run] [--backup]\n"
            "mode: mma_e3m4_a|b|ab  mma_e0m3_a|b|ab  cvt_e3m4  cvt_e0m3\n",
            argv[0]);
    return 2;
  }

  PatchStats st = run_patch(path, mode_s, dry, backup, /*verbose=*/true);
  if (st.rc != 0) fprintf(stderr, "%s\n", st.message.c_str());
  return st.rc;
}
