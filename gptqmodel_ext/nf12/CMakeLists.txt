# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.26)

# WGMMA, tcgen05 and SM120 TMA/setmaxnreg require architecture-specific cubins.
# Runtime probes reject loaded images without those feature paths.
# Embed generic PTX for the highest target so newer GPUs can JIT.
set(NF12_CUDA_ARCHS "80;86;89;90;100;110;120" CACHE STRING "CUDA architectures to build for")
if(DEFINED ENV{NF12_CUDA_ARCHS} AND NOT "$ENV{NF12_CUDA_ARCHS}" STREQUAL "")
  set(NF12_CUDA_ARCHS "$ENV{NF12_CUDA_ARCHS}" CACHE STRING "CUDA architectures to build for" FORCE)
endif()

set(_nf12_archs "")
set(_nf12_max_arch 0)
foreach(_arch IN LISTS NF12_CUDA_ARCHS)
  if(NOT _arch MATCHES "^[0-9]+$")
    message(FATAL_ERROR "NF12_CUDA_ARCHS entries must be plain numbers like 120, got '${_arch}'")
  endif()
  if(_arch LESS 80)
    message(FATAL_ERROR "NF12 requires SM80 or newer, got ${_arch}")
  endif()
  if(_arch MATCHES "^(90|100|103|110|120|121)$")
    list(APPEND _nf12_archs "${_arch}a-real")
  else()
    list(APPEND _nf12_archs "${_arch}-real")
  endif()
  if(_arch GREATER _nf12_max_arch)
    set(_nf12_max_arch ${_arch})
  endif()
endforeach()
list(APPEND _nf12_archs "${_nf12_max_arch}-virtual")
set(CMAKE_CUDA_ARCHITECTURES "${_nf12_archs}")

option(NF12_BUILD_NATIVE "Build native libraries during installation" ON)
if(NF12_BUILD_NATIVE)
  project(nf12 VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
  project(nf12 VERSION 1.0.0 LANGUAGES NONE)
endif()

# Installed wheels retain the sources for runtime builds without a checkout.
install(FILES CMakeLists.txt DESTINATION nf12/_src)
install(DIRECTORY csrc include DESTINATION nf12/_src
  FILES_MATCHING PATTERN "*.cu" PATTERN "*.cpp" PATTERN "*.cuh"
  PATTERN "*.h" PATTERN "*.map")
if(NOT NF12_BUILD_NATIVE)
  return()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)
message(STATUS "nf12: CUDA ${CUDAToolkit_VERSION}, archs ${CMAKE_CUDA_ARCHITECTURES}")

# Release wheels bundle libcudart with auditwheel.
if(NF12_CUDA_RUNTIME_LICENSE)
  install(FILES "${NF12_CUDA_RUNTIME_LICENSE}" DESTINATION nf12/licenses
    RENAME NVIDIA_CUDA_RUNTIME.txt)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

option(NF12_BUILD_TORCH "Build the torch extension (libnf12_ad_torch.so)" ON)

# --use_fast_math would violate the numerical contract in SPEC.md section 6.
# NVCC's warning policy does not cover the host compiler or PTX assembler.
set(NF12_CUDA_FLAGS -O3 -lineinfo --expt-relaxed-constexpr -Xfatbin=-compress-all
  -Xcompiler=-Werror -Xptxas=--warning-as-error)

file(GLOB_RECURSE NF12_KERNEL_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/codec/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/linear/*.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/*.cu")

add_library(nf12 SHARED
  csrc/api.cu
  csrc/device_info.cu
  ${NF12_KERNEL_SOURCES})
target_include_directories(nf12
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_compile_options(nf12 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NF12_CUDA_FLAGS}>)
target_link_libraries(nf12 PUBLIC CUDA::cudart)
# nf12.h has no visibility attributes, so the version script limits exports.
# --no-undefined catches missing kernel symbols at link time.
target_link_options(nf12 PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/csrc/nf12.map"
  "LINKER:--no-undefined")
set_target_properties(nf12 PROPERTIES
  LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/nf12.map"
  CUDA_SEPARABLE_COMPILATION OFF
  BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
  OUTPUT_NAME nf12_ad)

install(TARGETS nf12 LIBRARY DESTINATION nf12)

# torch.ops.load_library uses the stable ABI without a Python ABI dependency.
if(NF12_BUILD_TORCH)
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE NF12_TORCH_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  message(STATUS "nf12: torch at ${NF12_TORCH_DIR}")
  # Isolated builds can locate torch in a different temporary environment each time.
  unset(NF12_TORCH_CPU_LIB CACHE)
  unset(NF12_TORCH_CUDA_LIB CACHE)
  find_library(NF12_TORCH_CPU_LIB torch_cpu PATHS "${NF12_TORCH_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
  find_library(NF12_TORCH_CUDA_LIB torch_cuda PATHS "${NF12_TORCH_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

  file(GLOB NF12_TORCH_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/csrc/torch/*.cpp")
  add_library(nf12_torch SHARED ${NF12_TORCH_SOURCES} csrc/torch/fused_decode.cu)
  target_include_directories(nf12_torch SYSTEM PRIVATE "${NF12_TORCH_DIR}/include")
  target_include_directories(nf12_torch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/csrc")
  target_compile_options(nf12_torch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NF12_CUDA_FLAGS}>)
  # Restrict C-shim APIs to those available since torch 2.10.
  target_compile_definitions(nf12_torch PRIVATE USE_CUDA TORCH_TARGET_VERSION=0x020A000000000000ULL)
  target_link_libraries(nf12_torch PRIVATE nf12 "${NF12_TORCH_CPU_LIB}" "${NF12_TORCH_CUDA_LIB}")
  set_target_properties(nf12_torch PROPERTIES
    OUTPUT_NAME nf12_ad_torch
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    CUDA_SEPARABLE_COMPILATION OFF
    VISIBILITY_INLINES_HIDDEN ON
    BUILD_RPATH "$ORIGIN;${NF12_TORCH_DIR}/lib"
    INSTALL_RPATH "$ORIGIN")
  install(TARGETS nf12_torch LIBRARY DESTINATION nf12)
endif()
