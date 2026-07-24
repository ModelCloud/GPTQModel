// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#include <nvrtc.h>
#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

decltype(&::nvrtcAddNameExpression) p_nvrtcAddNameExpression;
decltype(&::nvrtcCompileProgram) p_nvrtcCompileProgram;
decltype(&::nvrtcCreateProgram) p_nvrtcCreateProgram;
decltype(&::nvrtcDestroyProgram) p_nvrtcDestroyProgram;
decltype(&::nvrtcGetCUBIN) p_nvrtcGetCUBIN;
decltype(&::nvrtcGetCUBINSize) p_nvrtcGetCUBINSize;
decltype(&::nvrtcGetErrorString) p_nvrtcGetErrorString;
decltype(&::nvrtcGetPTX) p_nvrtcGetPTX;
decltype(&::nvrtcGetPTXSize) p_nvrtcGetPTXSize;
decltype(&::nvrtcGetProgramLog) p_nvrtcGetProgramLog;
decltype(&::nvrtcGetProgramLogSize) p_nvrtcGetProgramLogSize;

#define nvrtcAddNameExpression p_nvrtcAddNameExpression
#define nvrtcCompileProgram p_nvrtcCompileProgram
#define nvrtcCreateProgram p_nvrtcCreateProgram
#define nvrtcDestroyProgram p_nvrtcDestroyProgram
#define nvrtcGetCUBIN p_nvrtcGetCUBIN
#define nvrtcGetCUBINSize p_nvrtcGetCUBINSize
#define nvrtcGetErrorString p_nvrtcGetErrorString
#define nvrtcGetPTX p_nvrtcGetPTX
#define nvrtcGetPTXSize p_nvrtcGetPTXSize
#define nvrtcGetProgramLog p_nvrtcGetProgramLog
#define nvrtcGetProgramLogSize p_nvrtcGetProgramLogSize

void die(const std::string &msg) {
  std::fprintf(stderr, "nvrtc_compile: %s\n", msg.c_str());
  std::exit(2);
}

template <typename T>
void load_symbol(void *library, T &symbol, const char *name) {
  symbol = reinterpret_cast<T>(dlsym(library, name));
  if (symbol == nullptr) die(std::string("cannot load ") + name + ": " + dlerror());
}

void load_nvrtc(const std::string &configured_path) {
  void *library = dlopen(configured_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr)
    die("cannot load NVRTC from " + configured_path + ": " + dlerror());
  load_symbol(library, p_nvrtcAddNameExpression, "nvrtcAddNameExpression");
  load_symbol(library, p_nvrtcCompileProgram, "nvrtcCompileProgram");
  load_symbol(library, p_nvrtcCreateProgram, "nvrtcCreateProgram");
  load_symbol(library, p_nvrtcDestroyProgram, "nvrtcDestroyProgram");
  load_symbol(library, p_nvrtcGetCUBIN, "nvrtcGetCUBIN");
  load_symbol(library, p_nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
  load_symbol(library, p_nvrtcGetErrorString, "nvrtcGetErrorString");
  load_symbol(library, p_nvrtcGetPTX, "nvrtcGetPTX");
  load_symbol(library, p_nvrtcGetPTXSize, "nvrtcGetPTXSize");
  load_symbol(library, p_nvrtcGetProgramLog, "nvrtcGetProgramLog");
  load_symbol(library, p_nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
}

std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void write_file(const std::string &path, const void *data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  if (!f) die("cannot write " + path);
  f.write(static_cast<const char *>(data), size);
}

#define NVRTC_CHECK(call)                                            \
  do {                                                               \
    nvrtcResult _r = (call);                                         \
    if (_r != NVRTC_SUCCESS) {                                       \
      die(std::string(#call " failed: ") + nvrtcGetErrorString(_r)); \
    }                                                                \
  } while (0)

struct Args {
  std::string nvrtc_path;
  std::string input;
  std::string output;
  std::string log_path;
  bool emit_ptx = false;
  std::vector<std::string> name_exprs;
  std::vector<std::pair<std::string, std::string>> headers; // name -> file path
  std::vector<std::string> nvrtc_flags;
};

Args parse_args(int argc, char **argv) {
  Args a;
  int i = 1;
  auto need = [&](const char *opt) {
    if (i + 1 >= argc) die(std::string("missing value for ") + opt);
    return std::string(argv[++i]);
  };
  for (; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--") {
      ++i;
      break;
    } else if (s == "--nvrtc-path") a.nvrtc_path = need("--nvrtc-path");
    else if (s == "-i" || s == "--input") a.input = need("--input");
    else if (s == "-o" || s == "--output") a.output = need("--output");
    else if (s == "--log") a.log_path = need("--log");
    else if (s == "--ptx") a.emit_ptx = true;
    else if (s == "--name-expression") a.name_exprs.push_back(need("--name-expression"));
    else if (s == "--header") {
      std::string v = need("--header");
      auto eq = v.find('=');
      if (eq == std::string::npos) die("--header expects NAME=PATH");
      a.headers.emplace_back(v.substr(0, eq), v.substr(eq + 1));
    } else if (s == "-h" || s == "--help") {
      std::printf(
          "Usage: nvrtc_compile --input X.cu --output Y.cubin "
          "[--nvrtc-path libnvrtc.so] [--ptx] [--log path] "
          "[--name-expression EXPR ...] [--header NAME=PATH ...] "
          "-- <nvrtc flag> ...\n");
      std::exit(0);
    } else die("unknown option: " + s);
  }
  for (; i < argc; ++i)
    a.nvrtc_flags.emplace_back(argv[i]);
  if (a.nvrtc_path.empty()) die("--nvrtc-path is required");
  if (a.input.empty()) die("--input is required");
  if (a.output.empty()) die("--output is required");
  return a;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  load_nvrtc(args.nvrtc_path);

  std::string src = read_file(args.input);

  std::vector<std::string> header_contents;
  header_contents.reserve(args.headers.size());
  std::vector<const char *> header_names_c;
  std::vector<const char *> header_sources_c;
  header_names_c.reserve(args.headers.size());
  header_sources_c.reserve(args.headers.size());
  for (auto &[name, path] : args.headers) {
    header_contents.push_back(read_file(path));
    header_names_c.push_back(name.c_str());
    header_sources_c.push_back(header_contents.back().c_str());
  }

  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(
      &prog, src.c_str(), args.input.c_str(),
      static_cast<int>(args.headers.size()),
      header_sources_c.empty() ? nullptr : header_sources_c.data(),
      header_names_c.empty() ? nullptr : header_names_c.data()));

  for (const auto &e : args.name_exprs) {
    NVRTC_CHECK(nvrtcAddNameExpression(prog, e.c_str()));
  }

  std::vector<const char *> opts;
  opts.reserve(args.nvrtc_flags.size());
  for (const auto &f : args.nvrtc_flags)
    opts.push_back(f.c_str());

  nvrtcResult compile_r = nvrtcCompileProgram(
      prog, static_cast<int>(opts.size()), opts.empty() ? nullptr : opts.data());

  size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  std::string log(log_size, '\0');
  if (log_size > 0) nvrtcGetProgramLog(prog, log.data());
  if (!log.empty() && log.back() == '\0') log.pop_back();

  if (!log.empty()) std::fwrite(log.data(), 1, log.size(), stderr);
  if (!args.log_path.empty()) write_file(args.log_path, log.data(), log.size());

  if (compile_r != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&prog);
    std::fprintf(stderr, "\nnvrtc_compile: compile failed: %s\n",
                 nvrtcGetErrorString(compile_r));
    return 1;
  }

  size_t out_size = 0;
  if (args.emit_ptx) {
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &out_size));
    std::string buf(out_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, buf.data()));
    write_file(args.output, buf.data(), out_size);
  } else {
    NVRTC_CHECK(nvrtcGetCUBINSize(prog, &out_size));
    if (out_size == 0) {
      nvrtcDestroyProgram(&prog);
      std::fprintf(stderr, "nvrtc_compile: empty CUBIN "
                           "(missing --gpu-architecture=sm_XX?)\n");
      return 1;
    }
    std::string buf(out_size, '\0');
    NVRTC_CHECK(nvrtcGetCUBIN(prog, buf.data()));
    write_file(args.output, buf.data(), out_size);
  }

  nvrtcDestroyProgram(&prog);
  return 0;
}
