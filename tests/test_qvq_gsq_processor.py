import io
import threading

import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import QVQConfig
from test_qvq_lifecycle import _prepared_calibration, _YaqaQModel


def test_gsq_processor_collect_fit_install_reload_cpu():
    """Actual full toy-model Fisher backward, quantizer and public backend."""
    torch.manual_seed(7)
    rows = [{"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
             "attention_mask": torch.ones(2, 3, dtype=torch.long)}]
    cfg = QVQConfig(bits=2.5, format="qvq_v2b2_p32", device="cpu", offload_to_disk=False,
                    yaqa={"minimum_sequences": 2}, gsq={"enabled": True, "steps": 4, "candidates": 3})
    qmodel = _YaqaQModel(cfg)
    processor = QVQProcessor(tokenizer=None, qcfg=cfg, calibration=rows, yaqa_calibration=rows,
                             prepare_dataset_func=_prepared_calibration, calibration_concat_size=None,
                             calibration_sort=None, batch_size=2)
    processor.prepare_yaqa(qmodel)
    module = qmodel.model.model.layers[0].proj
    named = NamedModule(module, name="proj", full_name="model.layers.0.proj", layer_index=0)
    processor.preprocess(named)
    source = qmodel.model.embed(rows[0]["input_ids"]).detach()
    processor._mask_tls = threading.local()
    processor._mask_tls.value = rows[0]["attention_mask"].bool()
    processor._set_current_batch_index(0)
    processor.pre_process_fwd_hook("proj")(module, (source,), module(source))
    processor.process(named, device=torch.device("cpu"))
    diag = processor.log[-1]["gsq"]
    assert diag["after"] <= diag["before"]
    assert diag["steps"] == 4
    assert named.full_name not in processor._yaqa_input_hessians
    live = processor.submodule_finalize(named, qmodel)
    buffer = io.BytesIO()
    torch.save(live.state_dict(), buffer)
    buffer.seek(0)
    reloaded = QVQLinear(bits=2.5, in_features=16, out_features=16, dtype=torch.float32,
                         tensors=torch.load(buffer, weights_only=True), bank_count=2, v2b2_p32=True).eval()
    torch.testing.assert_close(reloaded(source), live(source), atol=0, rtol=0)
    logits = qmodel.model(rows[0]["input_ids"], rows[0]["attention_mask"]).logits
    assert torch.isfinite(logits).all()
