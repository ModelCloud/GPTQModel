// Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
// SPDX-License-Identifier: Apache-2.0
//
// This file is vendored from https://github.com/inclusionAI/humming and used under
// the terms of the Apache License, Version 2.0.
// See gptqmodel/humming/LICENSE for the full license text.

#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define SHT_NOBITS 8
typedef uint64_t Elf64_Addr;
typedef uint64_t Elf64_Off;
typedef uint16_t Elf64_Half;
typedef uint32_t Elf64_Word;

#pragma pack(push, 1)
struct Elf64_Ehdr {
  unsigned char e_ident[16];
  Elf64_Half e_type;
  Elf64_Half e_machine;
  Elf64_Word e_version;
  Elf64_Addr e_entry;
  Elf64_Off e_phoff;
  Elf64_Off e_shoff;
  Elf64_Word e_flags;
  Elf64_Half e_ehsize;
  Elf64_Half e_phentsize;
  Elf64_Half e_phnum;
  Elf64_Half e_shentsize;
  Elf64_Half e_shnum;
  Elf64_Half e_shstrndx;
};

struct Elf64_Shdr {
  Elf64_Word sh_name;
  Elf64_Word sh_type;
  uint64_t sh_flags;
  Elf64_Addr sh_addr;
  Elf64_Off sh_offset;
  uint64_t sh_size;
  Elf64_Word sh_link;
  Elf64_Word sh_info;
  uint64_t sh_addralign;
  uint64_t sh_entsize;
};

struct Elf64_Sym {
  Elf64_Word st_name;
  unsigned char st_info;
  unsigned char st_other;
  Elf64_Half st_shndx;
  Elf64_Addr st_value;
  uint64_t st_size;
};
#pragma pack(pop)

class CubinReader {
private:
  std::map<std::string, uint64_t> symbolOffsets;
  std::vector<char> data;

public:
  CubinReader(const std::string &path) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) return;

    fs.seekg(0, std::ios::end);
    data.resize(static_cast<size_t>(fs.tellg()));
    fs.seekg(0);
    fs.read(data.data(), data.size());
    if (data.size() < sizeof(Elf64_Ehdr)) return;

    const Elf64_Ehdr *ehdr = reinterpret_cast<const Elf64_Ehdr *>(data.data());
    const Elf64_Shdr *shdrs = reinterpret_cast<const Elf64_Shdr *>(data.data() + ehdr->e_shoff);

    for (int i = 0; i < ehdr->e_shnum; ++i) {
      if (shdrs[i].sh_type == 2) {
        readSymbolTable(shdrs, ehdr->e_shnum, i);
      }
    }
  }

  uint32_t getUint32(const std::string &name) {
    auto it = symbolOffsets.find(name);
    if (it == symbolOffsets.end()) {
      throw std::runtime_error("cubin symbol not found: " + name);
    }
    if (it->second == 0ULL) return 0;

    uint32_t val = 0;
    std::memcpy(&val, data.data() + it->second, sizeof(uint32_t));
    return val;
  }

  bool getBool(const std::string &name) {
    return getUint32(name) > 0;
  }

  void readSymbolTable(const Elf64_Shdr *shdrs, int num_shdrs, int symIdx) {
    const Elf64_Shdr &symtab = shdrs[symIdx];
    const Elf64_Shdr &strtab = shdrs[symtab.sh_link];

    const char *names = data.data() + strtab.sh_offset;
    int count = symtab.sh_size / symtab.sh_entsize;
    const Elf64_Sym *syms = reinterpret_cast<const Elf64_Sym *>(data.data() + symtab.sh_offset);

    for (int i = 0; i < count; ++i) {
      const Elf64_Sym &sym = syms[i];
      if (sym.st_name == 0 || sym.st_name >= strtab.sh_size) continue;
      if (sym.st_shndx >= num_shdrs) continue;
      std::string sName = names + sym.st_name;

      const auto &shdr = shdrs[sym.st_shndx];
      if (shdr.sh_type == SHT_NOBITS) {
        symbolOffsets[sName] = 0;
      } else {
        uint64_t absOffset = shdr.sh_offset + sym.st_value;
        symbolOffsets[sName] = absOffset;
      }
    }
  }
};
