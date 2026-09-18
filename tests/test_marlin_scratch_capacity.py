# SPDX-License-Identifier: Apache-2.0
"""Execute the real shared C++ sizing helper without a CUDA device."""
from pathlib import Path
import shutil
import subprocess

import pytest
import torch


def test_shared_cpp_capacity_and_overflow(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler required for shared Marlin sizing check")
    root = Path(__file__).resolve().parents[1]
    torch_root = Path(torch.__file__).resolve().parent
    source = tmp_path / "capacity.cpp"
    binary = tmp_path / "capacity"
    source.write_text(r'''
#include "marlin_scratch.cuh"
#include <cassert>
#include <limits>

int main() {
  const int64_t rows[] = {0,1,8,16,17,32,64,128,512,2048};
  const int64_t expected[] = {0,540672,540672,540672,1081344,1081344,
                             2162688,2162688,2162688,2162688};
  for (int i=0; i<10; ++i) {
    const auto both = marlin::marlin_scratch_sizes_checked(rows[i],28672,132,256,true,true);
    assert(std::get<0>(both)==expected[i]);
    assert(std::get<1>(both)==rows[i]*28672);
    const auto off = marlin::marlin_scratch_sizes_checked(rows[i],28672,132,256,false,false);
    assert(std::get<0>(off)==0 && std::get<1>(off)==0);
  }
  const auto padded = marlin::marlin_scratch_sizes_checked(17,320,132,256,true,true);
  assert(std::get<1>(padded)==5440);
  const auto wide = marlin::marlin_scratch_sizes_checked(100000,100000,132,256,true,true);
  assert(std::get<1>(wide)==10000000000LL);
  assert(marlin::checked_scratch_mul(std::get<1>(wide),2,"bytes")==20000000000LL);
  bool negative=false, overflow=false, bytes=false;
  try { marlin::marlin_scratch_sizes_checked(-1,320,132,256,true,true); }
  catch(const c10::Error&) { negative=true; }
  try { marlin::marlin_scratch_sizes_checked(2,std::numeric_limits<int64_t>::max(),132,256,false,true); }
  catch(const c10::Error&) { overflow=true; }
  try { marlin::checked_scratch_mul(std::numeric_limits<int64_t>::max(),4,"bytes"); }
  catch(const c10::Error&) { bytes=true; }
  assert(negative && overflow && bytes);
}
''')
    command = [compiler, "-std=c++17", "-O0",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               "-I" + str(root / "gptqmodel_ext/marlin"),
               "-I" + str(torch_root / "include"), str(source),
               "-L" + str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
               "-lc10", "-o", str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
