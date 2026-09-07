// SPDX-License-Identifier: Apache-2.0
#include "qvq_gfx950_abi.h"
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <new>

struct qvq_gfx950_plan {
    hipModule_t module;
    hipFunction_t function;
    qvq_gfx950_spec spec;
    int device;
};

static thread_local char last_error[512];
static int fail(const char* message) {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
    return 1;
}
static int checked(hipError_t error) {
    if (error == hipSuccess) return 0;
    return fail(hipGetErrorString(error));
}

extern "C" int qvq_gfx950_abi_version() { return QVQ_GFX950_ABI_VERSION; }
extern "C" const char* qvq_gfx950_last_error() { return last_error; }
extern "C" int qvq_gfx950_device_supported(int device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess) return 0;
    return std::strncmp(props.gcnArchName, "gfx950", 6) == 0 &&
        (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

extern "C" int qvq_gfx950_prepare(const qvq_gfx950_spec* spec,
    const void* image, size_t image_size, const char* symbol, int device,
    void* stream, qvq_gfx950_plan** result) {
    if (!result) return fail("null result");
    *result = nullptr;
    if (!spec || !image || image_size < 64 || !symbol || !symbol[0])
        return fail("invalid artifact");
    if (spec->abi_version != QVQ_GFX950_ABI_VERSION ||
        spec->operation_version != QVQ_GFX950_OPERATION_VERSION)
        return fail("incompatible QVQ ABI/operation version");
    if (!spec->m || !spec->k || !spec->n || spec->k % 16 || spec->n % 16 ||
        spec->transition_bits < 4 || spec->transition_bits > 8 ||
        (spec->transition_bits == 8 ? spec->bank_alt_id != 0 :
         spec->bank_alt_id < 1 || spec->bank_alt_id > 3) ||
        !spec->grid_x || !spec->threads || spec->threads > 1024 ||
        spec->threads % 64)
        return fail("invalid QVQ geometry or launch descriptor");
    int current;
    if (checked(hipGetDevice(&current))) return 1;
    if (current != device || !qvq_gfx950_device_supported(device))
        return fail("QVQ plan requires current gfx950 device");
    hipDeviceProp_t properties{};
    if (checked(hipGetDeviceProperties(&properties, device))) return 1;
    if (spec->shared_bytes > properties.sharedMemPerBlock)
        return fail("QVQ shared memory exceeds device block limit");
    hipStreamCaptureStatus capture;
    if (checked(hipStreamIsCapturing(static_cast<hipStream_t>(stream), &capture))) return 1;
    if (capture != hipStreamCaptureStatusNone) return fail("prepare must precede capture");
    auto* plan = new (std::nothrow) qvq_gfx950_plan{};
    if (!plan) return fail("host allocation failed");
    plan->device = device;
    plan->spec = *spec;
    if (checked(hipModuleLoadData(&plan->module, image))) { delete plan; return 1; }
    if (checked(hipModuleGetFunction(&plan->function, plan->module, symbol))) {
        // Preserve the symbol-lookup diagnostic if best-effort cleanup fails.
        const hipError_t cleanup_status = hipModuleUnload(plan->module);
        (void)cleanup_status;
        delete plan;
        return 1;
    }
    *result = plan;
    return 0;
}

extern "C" int qvq_gfx950_execute(const qvq_gfx950_plan* plan,
    const void* input, const void* window, const void* levels, const void* banks,
    void* output, void* stream) {
    if (!plan || !input || !window || !levels || !banks || !output)
        return fail("null QVQ plan or buffer");
    int current;
    if (checked(hipGetDevice(&current))) return 1;
    if (current != plan->device) return fail("QVQ device differs from prepared device");
    // Triton appends global/profile scratch pointers even when both sizes
    // are zero. Export rejects artifacts requiring either allocation.
    void* scratch = nullptr;
    void* args[] = {&input, &window, &levels, &banks, &output, &scratch, &scratch};
    return checked(hipModuleLaunchKernel(plan->function, plan->spec.grid_x, 1, 1,
        plan->spec.threads, 1, 1, plan->spec.shared_bytes,
        static_cast<hipStream_t>(stream), args, nullptr));
}

extern "C" int qvq_gfx950_destroy(qvq_gfx950_plan* plan) {
    if (!plan) return 0;
    int current;
    if (checked(hipGetDevice(&current))) return 1;
    if (current != plan->device) return fail("destroy requires prepared device");
    if (checked(hipModuleUnload(plan->module))) return 1;
    delete plan;
    return 0;
}
