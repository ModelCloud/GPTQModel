"""Laguna single-GPU benchmark compatibility adapter for SGLang."""

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as _VocabParallelEmbedding,
)
from sglang.srt.models.laguna import LagunaForCausalLM as _MainLagunaForCausalLM

import laguna_benchmark_common as common
from laguna_gptq_embedding import (
    EmbeddingQuantConfigProxy,
    GPTQEmbeddingW8G128MethodMixin,
)


class _GPTQEmbeddingW8G128Method(GPTQEmbeddingW8G128MethodMixin, QuantizeMethodBase):
    pass


class _BenchmarkVocabParallelEmbedding(_VocabParallelEmbedding):
    _quant_config = None

    def __init__(self, *args, quant_config=None, prefix: str = "", **kwargs):
        quant_config = quant_config or self._quant_config
        if common.USES_QUANTIZED_ENDPOINTS and prefix.endswith("embed_tokens"):
            quant_config = EmbeddingQuantConfigProxy(
                quant_config, _GPTQEmbeddingW8G128Method()
            )
        super().__init__(*args, quant_config=quant_config, prefix=prefix, **kwargs)


class LagunaForCausalLM(_MainLagunaForCausalLM):
    """Enable benchmark-local packed W8 embedding support when requested."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        from sglang.srt.models import laguna as laguna_module

        original_embedding = laguna_module.VocabParallelEmbedding
        _BenchmarkVocabParallelEmbedding._quant_config = quant_config
        if common.USES_QUANTIZED_ENDPOINTS:
            laguna_module.VocabParallelEmbedding = _BenchmarkVocabParallelEmbedding
        try:
            super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        finally:
            laguna_module.VocabParallelEmbedding = original_embedding
            _BenchmarkVocabParallelEmbedding._quant_config = None


EntryClass = LagunaForCausalLM
