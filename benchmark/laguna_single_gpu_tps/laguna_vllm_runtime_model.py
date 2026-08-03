"""Laguna single-GPU benchmark compatibility adapter for vLLM."""

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as _VocabParallelEmbedding,
)
from vllm.model_executor.models.laguna import (
    LagunaForCausalLM as _MainLagunaForCausalLM,
)

import laguna_benchmark_common as common
from laguna_gptq_embedding import (
    EmbeddingQuantConfigProxy,
    GPTQEmbeddingW8G128MethodMixin,
)


class _GPTQEmbeddingW8G128Method(GPTQEmbeddingW8G128MethodMixin, QuantizeMethodBase):
    pass


class _BenchmarkVocabParallelEmbedding(_VocabParallelEmbedding):
    def __init__(self, *args, quant_config=None, prefix: str = "", **kwargs):
        if common.USES_QUANTIZED_ENDPOINTS and prefix.endswith("embed_tokens"):
            quant_config = EmbeddingQuantConfigProxy(
                quant_config, _GPTQEmbeddingW8G128Method()
            )
        super().__init__(*args, quant_config=quant_config, prefix=prefix, **kwargs)


class LagunaForCausalLM(_MainLagunaForCausalLM):
    """Enable benchmark-local packed W8 embedding support when requested."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        from vllm.model_executor.models import laguna as laguna_module

        original_embedding = laguna_module.VocabParallelEmbedding
        if common.USES_QUANTIZED_ENDPOINTS:
            laguna_module.VocabParallelEmbedding = _BenchmarkVocabParallelEmbedding
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            laguna_module.VocabParallelEmbedding = original_embedding
