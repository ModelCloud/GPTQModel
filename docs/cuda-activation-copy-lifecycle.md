# CUDA activation copy lifecycle

`CudaEventActivationBuffer.capture_async` copies a forward activation into
pinned host memory on a dedicated CUDA stream. The producer stream finishes
writing the activation before the copy begins. The device allocation must
remain unavailable for reuse until the copy finishes, even if the forward hook
drops its last tensor reference immediately. `recycle` must likewise wait
before returning a host buffer to a caller-provided pool.

```mermaid
flowchart TD
    A["capture_async receives activation"] --> B{"Activation on buffer device?"}
    B -- No --> X["Raise ValueError"]
    B -- Yes --> C["Make contiguous if needed and allocate pinned host tensor"]
    C --> D["copy_stream.wait_stream waits for producer"]
    D --> E["host.copy_ enqueues device to host copy"]
    E --> F["activation.record_stream protects device storage"]
    F --> G["Record completion event and return ActivationPacket"]
    G --> H{"Consumer action"}
    H -- "drain wait=True" --> I["Synchronize event before host read"]
    H -- "drain wait=False" --> J["Caller synchronizes event before host read"]
    H -- recycle --> K["Synchronize event before host pool reclaims tensor"]
    classDef failure fill:#ffe8e8,stroke:#b30000,color:#111;
    class X failure;
```

| Storage | Before | Current |
| --- | --- | --- |
| Device activation | `capture_async` queued the side-stream copy without informing the caching allocator (`d3253c2:gptqmodel/utils/cuda_activation_buffer.py:110-115`). The producer could release and overwrite the same storage before the copy read it. | `activation.record_stream(copy_stream)` protects the allocation through the queued copy (`gptqmodel/utils/cuda_activation_buffer.py:113-120`). |
| Pinned host buffer | `recycle` called the host reclaimer even when `capture_async(enqueue=False)` or `drain(wait=False)` returned an in-flight packet (`d3253c2:gptqmodel/utils/cuda_activation_buffer.py:147-152`). | `recycle` synchronizes the packet event before passing its tensor to the pool (`gptqmodel/utils/cuda_activation_buffer.py:152-162`). |

The CUDA regression tests delay the copy stream, release the original device
tensor, and allocate a replacement. Before the fix, the host received the
replacement's `9.0` values instead of the original `3.0` values on NVIDIA
GB10; the reclaimer also observed an incomplete event. Both conditions pass
after the fix, including a noncontiguous activation that creates a temporary
contiguous copy.

`record_stream` protects against allocator reuse. Callers must still avoid
modifying an activation in place while its copy is pending and must wait for
the packet event before reading the host tensor. See PyTorch's
[`record_stream` contract](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)
and [CUDA stream guidance](https://docs.pytorch.org/docs/stable/notes/cuda.html).

Introduced in #3646.
