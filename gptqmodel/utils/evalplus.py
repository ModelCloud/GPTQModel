import types

from tokenicer import Tokenicer
from transformers import PreTrainedModel


def patch_strip(self, *args, **kwargs):
    return self.config.name_or_path.strip(*args, **kwargs)

def patch_tostring(self):
    return self.config.name_or_path

def patch_evalplus(model):
    if isinstance(model, str):
        return

    assert model.tokenizer, "model must have a tokenizer to use evalplus!"
    model.strip = types.MethodType(patch_strip, model)
    model.__str__ = types.MethodType(patch_tostring, model)

    import torch
    from evalplus.provider.base import DecoderBase
    from evalplus.provider.gptqmodel import GPTQModelDecoder
    from evalplus.provider.utility import extra_eos_for_direct_completion
    from gptqmodel.models import BaseGPTQModel

    from .. import GPTQModel

    class PatchedGPTQModelDecoder(DecoderBase):
        def __init__(
                self,
                name: str,
                dataset: str,
                gptqmodel_backend: str = 'auto',
                force_base_prompt: bool = False,
                **kwargs,
        ):

            super(GPTQModelDecoder, self).__init__(name=name, **kwargs)

            if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
                device = torch.device("mps")
            elif hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
                device = torch.device("xpu")
            elif hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            self.device = device

            kwargs = {
                "model_id_or_path": name,
                "trust_remote_code": self.trust_remote_code,
                "backend": gptqmodel_backend,
                "device": device
            }
            self.skip_special_tokens = True
            self.force_base_prompt = force_base_prompt
            if isinstance(name, BaseGPTQModel):
                self.model = name
                self.tokenizer = self.model.tokenizer
            elif isinstance(name, PreTrainedModel):
                self.model = name
                self.tokenizer = Tokenicer.load(name.config.name_or_path, trust_remote_code=self.trust_remote_code)
            else:
                self.tokenizer = Tokenicer.load(name, trust_remote_code=self.trust_remote_code)
                self.model = GPTQModel.load(**kwargs)
                self.model = self.model.to(self.device)
            if self.is_direct_completion():  # no chat template
                self.eos += extra_eos_for_direct_completion(dataset)
            else:  # with chat template
                self.eos += ["\n```\n"]

    GPTQModelDecoder.__init__ = PatchedGPTQModelDecoder.__init__
