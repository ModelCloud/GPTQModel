import torch
from parameterized import parameterized

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig


class _ClipTestAWQProcessor(AWQProcessor):
    def __init__(self, qcfg: QuantizeConfig) -> None:
        super().__init__(
            tokenizer=None,
            qcfg=qcfg,
            calibration=None,
            prepare_dataset_func=None,
            calibration_concat_size=None,
            calibration_sort=None,
            batch_size=1,
            gptq_model=None,
            model=None,
            require_fwd=True,
            calculate_w_wq_diff=False,
            calibration_concat_separator=None,
        )

    def _module_forward(self, x, module, module_kwargs):
        return module(x)


def _legacy_clip(processor: AWQProcessor, w: torch.Tensor, input_feat: torch.Tensor):
    group_size = processor.qcfg.group_size if processor.qcfg.group_size > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    step_size = max(1, input_feat.shape[1] // 512)
    input_feat = input_feat[:, ::step_size]

    w = w.reshape(w.shape[0], 1, -1, group_size)
    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64
    assert w.shape[0] % oc_batch_size == 0
    best_max_val_all = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w_chunk = w[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]
        org_max_val = w_chunk.abs().amax(dim=-1, keepdim=True)
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w_chunk.device)
        org_out = (input_feat * w_chunk).sum(dim=-1)
        for i_s in range(int(0.5 * 20)):
            max_val = org_max_val * (1 - i_s / 20)
            min_val = -max_val
            cur_w = torch.clamp(w_chunk, min_val, max_val)
            q_w = processor.pseudo_quantize_tensor(cur_w)[0]
            cur_out = (input_feat * q_w).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)
    return torch.cat(best_max_val_all, dim=0).squeeze(1)


@parameterized.expand([
    ("cpu", "cpu"),
    ("cuda", "cuda:0"),
])
def test_awq_clip_consistency(device_name: str, device_str: str):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise AssertionError("CUDA is not available for clip consistency test")

    dtype = torch.float32 if device_name == "cpu" else torch.float16
    processor = _ClipTestAWQProcessor(QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=128))

    out_features = 256
    in_features = 3584
    w = torch.randn(out_features, in_features, dtype=dtype, device=device_str)
    tokens = 1024
    input_feat = torch.randn(tokens, in_features, dtype=dtype, device=device_str)

    # Compare the streaming implementation against the legacy tensor-per-iter path
    expected = _legacy_clip(processor, w.clone(), input_feat.clone())
    actual = processor._compute_best_clip(w, input_feat)

    tol = 1e-6 if dtype == torch.float32 else 1e-4
    assert torch.allclose(actual.cpu(), expected.cpu(), atol=tol, rtol=tol), \
        f"Inconsistent clip: max diff {(actual - expected).abs().max().item():.3e}"
