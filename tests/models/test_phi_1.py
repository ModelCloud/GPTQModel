from model_test import ModelTest

class TestPhi_1(ModelTest):
    NATIVE_MODEL_ID = "microsoft/phi-1"

    def test_phi_1(self):
        model, tokenizer = self.quantModel(self.NATIVE_MODEL_ID)
        reference_output = "I am in Paris and I like to eat pizza and pasta.\"\n    \"\"\"\n    words = s.split()\n    new_words = []\n    for word in words:\n        if word.lower() == \"paris\":\n            new_words.append(\"pizza\")\n        elif word.lower() == \"pasta\":\n            new_words.append(\"pasta\")\n        else:\n            new_words.append(word)\n    return " ".join\n"
        result = self.generate(model, tokenizer)

