# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Launch every shipped norm-rank configuration under Compute Sanitizer."""

import torch

from gptqmodel.quantization.qvq_codecs import pgc16_codebook_v2_bank
from gptqmodel.utils.qvq_cuda import _qvq_cuda_viterbi_v2_segment_grid_trusted_op


def main() -> None:
    op = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()
    for bits, bank_count, segment_steps in (
        (2.5, 2, 16),
        (2.5, 4, 32),
        (3.0, 2, 16),
        (3.0, 4, 32),
    ):
        generator = torch.Generator(device="cuda").manual_seed(
            20260840 + int(bits * 2) * 10 + bank_count
        )
        # Batch nine keeps W2.5/b2 on the norm-rank grid rather than its
        # small-batch cooperative sibling.
        sequences = torch.randn((9, 128, 2), generator=generator, device="cuda")
        codebooks = torch.stack(
            tuple(
                pgc16_codebook_v2_bank(bank, bits=bits, dtype=torch.float32)
                for bank in range(bank_count)
            )
        ).to(device="cuda", dtype=torch.float16)
        for constrained in (False, True):
            overlap = (
                torch.randint(
                    0,
                    1 << (16 - int(bits * 2)),
                    (sequences.shape[0],),
                    generator=generator,
                    device="cuda",
                    dtype=torch.int64,
                )
                if constrained
                else None
            )
            before = int(torch.ops.gptqmodel_qvq.norm_rank_grid_dispatch_count())
            op(sequences, codebooks, int(bits * 2), segment_steps, overlap, None)
            torch.cuda.synchronize()
            assert int(torch.ops.gptqmodel_qvq.norm_rank_grid_dispatch_count()) == before + 1
            print(
                f"RACECHECK-LAUNCH W{bits:g} b{bank_count} seg{segment_steps} "
                f"constrained={constrained}"
            )


if __name__ == "__main__":
    main()
