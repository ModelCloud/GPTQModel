import torch

from models.model_test import ModelTest


class FakeBatchEncoding(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 7

    def __init__(self):
        self.decode_calls = []
        self.batch_decode_calls = []

    def __call__(self, prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        assert prompt == "hello"
        return FakeBatchEncoding(torch.tensor([[101, 102]]))

    def decode(self, tokens, skip_special_tokens=True):
        self.decode_calls.append(
            {
                "tokens": tokens.tolist(),
                "skip_special_tokens": skip_special_tokens,
            }
        )
        return f"decoded:{tokens.tolist()}"

    def batch_decode(self, sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        self.batch_decode_calls.append(
            {
                "sequences": [seq.tolist() for seq in sequences],
                "skip_special_tokens": skip_special_tokens,
                "clean_up_tokenization_spaces": clean_up_tokenization_spaces,
            }
        )
        return [f"batch:{[seq.tolist() for seq in sequences]}"]


class FakeProcessor(FakeTokenizer):
    pass


class FakeModel:
    def __init__(self, generated):
        self.device = "cpu"
        self.generated = generated
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.generated


def test_generate_stable_with_limit_for_prompt_uses_deterministic_kwargs():
    tokenizer = FakeTokenizer()
    model = FakeModel(torch.tensor([[101, 102, 103, 104]]))

    output = ModelTest.generate_stable_with_limit(
        model,
        tokenizer,
        "hello",
        min_new_tokens=2,
        max_new_tokens=4,
        skip_special_tokens=False,
    )

    assert output == "decoded:[101, 102, 103, 104]"
    assert len(model.calls) == 1
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["num_beams"] == 1
    assert model.calls[0]["min_new_tokens"] == 2
    assert model.calls[0]["max_new_tokens"] == 4
    assert model.calls[0]["pad_token_id"] == tokenizer.eos_token_id
    assert model.calls[0]["eos_token_id"] == tokenizer.eos_token_id
    assert tokenizer.decode_calls == [
        {
            "tokens": [101, 102, 103, 104],
            "skip_special_tokens": False,
        }
    ]


def test_generate_stable_with_limit_for_prepared_inputs_batch_decodes_suffix():
    processor = FakeProcessor()
    prepared_inputs = FakeBatchEncoding(torch.tensor([[10, 11]]))
    model = FakeModel(torch.tensor([[10, 11, 21, 22]]))

    output = ModelTest.generate_stable_with_limit(
        model,
        processor,
        inputs=prepared_inputs,
        prompt=None,
        batch_decode=True,
        max_new_tokens=2,
        clean_up_tokenization_spaces=False,
    )

    assert output == "batch:[[21, 22]]"
    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].tolist() == [[10, 11]]
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["num_beams"] == 1
    assert processor.batch_decode_calls == [
        {
            "sequences": [[21, 22]],
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        }
    ]
