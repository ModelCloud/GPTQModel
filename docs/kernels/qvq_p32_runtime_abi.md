# QVQ P32 framework-neutral runtime ABI

The canonical Ampere P32 CUDA runtime lives under
`gptqmodel_ext/qvq/p32/`. It accepts caller-owned device buffers, workspace,
and a CUDA stream without depending on PyTorch, PJRT, XLA, or Zig.

Framework integrations own graph construction and tuning policy. They may use
native split reduction or request split partials so a graph compiler can keep
the reduction and neighboring operations visible. No ABI entry point allocates
device memory, changes the current stream, or synchronizes the device.

The public header is the canonical operation and launch contract. It defines
the mathematical operation version, C ABI version, kernel version, SM target,
tile geometry, supported transition-bit range, launch limits, and enum values.
The ABI and kernel versions are also exported by the shared object. Consumers
must include both values in transient autotune keys and reject incompatible
libraries during initialization.

The initial library target is deliberately SM80-only. Additional architecture
libraries must use distinct targets and runtime capability gates rather than a
single implicit device assumption.
