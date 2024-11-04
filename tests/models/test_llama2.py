from model_test import ModelTest
import torch

class TestLlama2(ModelTest):
    NATIVE_MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"

    def test_llama2(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = " Of course, I'd be happy to help! The Shanghai Natural History Museum is a fantastic place to visit, and I'm sure you'll have a great time there. Here are some tips to make the most of your visit:\n\n1. Plan Your Visit: The Shanghai Natural History Museum is open from 9:00 AM to 5:00 PM, Tuesday through Sunday. It's closed on Mondays. You can"
        result = self.generateChat(model, tokenizer)

