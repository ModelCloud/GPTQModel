#include <cuda_fp16.h>

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hadamard.h"
#include "hgemm.cuh"

#include "quant/quantize.cuh"
#include "quant/pack.cuh"
#include "quant/reconstruct.cuh"
#include "quant/hadamard.cuh"

#include "libtorch/linear.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("had_paley", &had_paley, "had_paley");
    m.def("had_paley2", &had_paley2, "had_paley2");

    m.def("quantize_tiles", &quantize_tiles, "quantize_tiles");
    m.def("pack_trellis", &pack_trellis, "pack_trellis");
    m.def("unpack_trellis", &unpack_trellis, "unpack_trellis");
    m.def("pack_signs", &pack_signs, "pack_signs");
    m.def("reconstruct", &reconstruct, "reconstruct");
    m.def("had_r_128", &had_r_128, "had_r_128");
    m.def("hgemm", &hgemm, "hgemm");

    #include "libtorch/linear_bc.h"
}
