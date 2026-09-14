"""Execute public calibrated FP8 quantization on real Llama block-0 QKV on a leased GPU."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.validate_qvq_gsq_layers import TARGETS, digest, write_json


def execute(output):
    provenance = json.loads((output / "provenance.json").read_text())
    if (output / "report.json").exists():
        raise ValueError("Execution already started; preserve it and prepare a fresh directory")
    for path, sha in provenance["files"].items():
        if digest(path) != sha:
            raise ValueError(f"Prepared source changed: {path}")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid or not os.environ.get("GPU_ALLOCATOR_LEASE_ID"):
        raise ValueError("Requires one exclusive GPU allocator UUID lease")
    for _ in range(3):
        inventory = subprocess.check_output([
            "nvidia-smi", "--id=" + uuid,
            "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip()
        fields = [item.strip() for item in inventory.split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[5]) or uuid in processes:
            raise RuntimeError("Idle GPU preflight failed: " + inventory)
        print("IDLE", inventory, flush=True)
        time.sleep(1)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gptqmodel.models.definitions.llama import LlamaQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.quantization.config import FP8Config
    from gptqmodel.utils.backend import BACKEND

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    inputs = Path(provenance['inputs'])
    source = json.loads((inputs/'provenance.json').read_text())
    documents = json.loads((inputs/'inputs.json').read_text())
    cfg = FP8Config(device=DEVICE.CUDA, offload_to_disk=False, gsq_calibration=True,
                    dynamic={r'-:model\.layers\.[1-9][0-9]*\.': {},
                             r'-:model\.layers\.0\.mlp\.': {},
                             r'-:model\.layers\.0\.self_attn\.o_proj$': {}},
                    gsq={'enabled': True, 'steps': 100, 'candidates': 3, 'seed': 7})
    write_json(output/'quantize_config.json', cfg.to_dict())
    report = {'state': 'loading', 'inventory': inventory, 'provenance': provenance,
              'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'layers': {}}
    write_json(output/'report.json', report)
    native = AutoModelForCausalLM.from_pretrained(source['dense'], dtype=torch.float32,
               attn_implementation='eager', local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(source['dense'], local_files_only=True)
    wrapper = LlamaQModel(model=native, quantized=False, quantize_config=cfg, tokenizer=tokenizer)
    calibration = [{'input_ids': row['input_ids'], 'attention_mask': [1]*len(row['input_ids']),
                    'fisher_sequence_weight': [provenance['train_source_weights'][row['source_name']]]}
                   for row in documents['train']]
    report['state'] = 'quantizing'
    write_json(output/'report.json', report)
    quant_log = wrapper.quantize(calibration=calibration, calibration_sort=None, batch_size=1,
                                 backend=BACKEND.FP8_TORCH)
    write_json(output/'quantization-log.json', quant_log)
    expected_tokens = sum(len(row['input_ids']) for row in documents['train'])
    expected_weighted = sum(len(row['input_ids'])*provenance['train_source_weights'][row['source_name']]
                            for row in documents['train'])
    entries = quant_log['fp8_gsq']
    if len(entries) != len(TARGETS):
        raise AssertionError('Expected one quantization record per QKV projection')
    for entry in entries:
        stats = entry['gsq']
        if stats['tokens'] != expected_tokens or abs(stats['weighted_tokens']-expected_weighted) > 1e-6:
            raise AssertionError(f'Calibration token weighting mismatch: {stats}')
    report['verified_tokens'] = expected_tokens
    report['verified_weighted_tokens'] = expected_weighted
    native = wrapper.model
    report['module_types'] = {name: type(native.get_submodule(name)).__name__ for name in TARGETS}
    write_json(output/'report.json', report)
    actual = {name for name, module in native.named_modules() if isinstance(module, TorchFP8Linear)}
    if actual != set(TARGETS):
        raise AssertionError(f'Unexpected quantized modules: {actual}')
    for name in TARGETS:
        packed = native.get_submodule(name)
        path = output/f'{name}.pt'
        torch.save(packed.state_dict(), path)
        restored = TorchFP8Linear(bits=8, group_size=-1, sym=True, desc_act=False,
                    in_features=packed.in_features, out_features=packed.out_features, bias=False)
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        fixture = torch.load(inputs/f'{name}.inputs.pt', weights_only=True)
        records = []
        with torch.no_grad():
            for index, x in enumerate(fixture['heldout']):
                output_tensor = restored(x.float()).float()
                reference = x.float() @ fixture['weight'].T
                error = (output_tensor.double()-reference.double()).square()
                records.append({'document': index, 'sse': error.sum().item(), 'elements': error.numel()})
        report['layers'][name] = {'payload_sha256': digest(path), 'rows': records,
            'heldout_mse': sum(v['sse'] for v in records)/sum(v['elements'] for v in records)}
        write_json(output/'report.json', report)
    report['state'] = 'complete'
    write_json(output/'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output.resolve())
