import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from gptqmodel.utils.cuda_graph import StaticCUDAGraphGreedyRunner


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="CUDA Graph decode regression test requires sm80 CUDA",
)
def test_static_cuda_graph_greedy_runner_matches_dense_fp16_and_reuses_cache():
    torch.manual_seed(1234)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=None,
    )
    model = Qwen3ForCausalLM(config).cuda().half().eval()
    example = torch.tensor([[1, 2, 3, 4, 5]], device="cuda", dtype=torch.long)

    runner = StaticCUDAGraphGreedyRunner(
        model,
        example,
        max_new_tokens=8,
        graph_warmup=2,
    )
    prompts = (example, torch.tensor([[7, 8, 9]], device="cuda", dtype=torch.long))
    with torch.inference_mode():
        for prompt in prompts:
            expected = model.generate(
                prompt,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
                max_new_tokens=6,
                cache_implementation="dynamic",
                disable_compile=True,
            )
            actual = runner.generate(prompt, max_new_tokens=6)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert torch.equal(actual, expected)

        first_eos = int(runner.generate(example, max_new_tokens=1)[0, -1].item())
        truncated = runner.generate(example, max_new_tokens=6, eos_token_id=first_eos)
        assert truncated.shape[1] == example.shape[1] + 1
        assert truncated[0, -1].item() == first_eos

        fixed_prefill_runner = StaticCUDAGraphGreedyRunner(
            model,
            example,
            max_new_tokens=8,
            graph_warmup=2,
            release_prefill_cache=False,
            capture_prefill=True,
        )
        fixed_prompts = (example, torch.tensor([[9, 8, 7, 6, 5]], device="cuda", dtype=torch.long))
        for fixed_prompt in fixed_prompts:
            expected = model.generate(
                fixed_prompt,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
                max_new_tokens=6,
                cache_implementation="dynamic",
                disable_compile=True,
            )
            assert torch.equal(fixed_prefill_runner.generate(fixed_prompt, max_new_tokens=6), expected)
            assert torch.equal(fixed_prefill_runner.generate(fixed_prompt, max_new_tokens=6), expected)
        with pytest.raises(ValueError, match="exactly 5 prompt tokens"):
            fixed_prefill_runner.generate(prompts[1], max_new_tokens=6)
