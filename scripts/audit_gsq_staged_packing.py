"""Audit real staged scalar weights against portable GPTQ packing and reload."""

import argparse
import json
from pathlib import Path

import torch

from gptqmodel import BACKEND
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


def audit(root):
    bits = json.loads((root/'report.json').read_text())['bits']
    if bits not in (2, 3, 4):
        raise ValueError('Packing audit requires W2, W3 or W4')
    payload = torch.load(root/'stages.pt', map_location='cpu', weights_only=True)
    records = {}
    for stage, result in payload['stages'].items():
        for key, weight in result['weights'].items():
            name = stage if key == 'weight' else key.removesuffix('.weight')
            scales = result['scales'][key]
            rows, columns = weight.shape
            groups = torch.arange(columns, dtype=torch.int32)//128
            linear = torch.nn.Linear(columns, rows, bias=False, device='meta')
            linear.weight = torch.nn.Parameter(weight, requires_grad=False)

            def container():
                return TorchLinear(bits=bits, group_size=128, sym=True, desc_act=False,
                                   in_features=columns, out_features=rows, bias=False, backend=BACKEND.TORCH)
            packed = container()
            packed.pack_original(linear, scales, torch.full_like(scales, 2**(bits-1)), groups)
            restored = container()
            restored.load_state_dict(packed.state_dict(), strict=True)
            codes, zeros = restored._unpack_continuous_codes()
            expected_codes = (weight/scales[:, groups]).round()+2**(bits-1)
            mismatch = int((codes.T != expected_codes).sum())
            if mismatch:
                raise AssertionError(f'{name}: {mismatch} assignment mismatches')
            decoded = (restored.scales.float()[groups]*(codes.float()-zeros.float()[groups])).T
            expected = (expected_codes-2**(bits-1))*restored.scales.float().T[:, groups]
            torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
            records[name] = dict(assignment_mismatches=mismatch, negative_scales=int((scales < 0).sum()),
                                 zero_stored_scales=int((restored.scales == 0).sum()),
                                 scale_storage_dtype=str(restored.scales.dtype),
                                 decoded_weight_mse=float((decoded-weight).square().mean()))
            print(name, records[name], flush=True)
    (root/'packing-audit.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
