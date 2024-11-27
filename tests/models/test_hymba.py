from model_test import ModelTest


class TestHymba(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Hymba-1.5B-Instruct/"  # "baichuan-inc/Baichuan2-7B-Chat"
    NATIVE_ARC_CHALLENGE_ACC = 0.4104
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.4317
    MODEL_MAX_LEN = 8192
    TRUST_REMOTE_CODE = True
    BATCH_SIZE = 6

    def test_hymba(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE,
                                           torch_dtype=self.TORCH_DTYPE)
        model.cuda()

        prompt = "5+5=?"

        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]

        # Apply chat template
        tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt").to('cuda')

        outputs = model.generate(input_ids=tokenized_chat, max_new_tokens=56)
        input_length = tokenized_chat.shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        print(f"Model response: {response}")

        self.assertTrue("10" in response)
