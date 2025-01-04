# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402

import requests  # noqa: E402


class TestModelInference(unittest.TestCase):
    def generate_request(self, port):
        url = f"http://127.0.0.1:{port}/generate_stream"
        headers = {'Content-Type': 'application/json'}
        payload = {
            "inputs": "What is Deep Learning?",
            "parameters": {"max_new_tokens": 20}
        }
        response = requests.post(url, json=payload, headers=headers, stream=True)
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    json_data = decoded_line[len("data: "):]
                    data = json.loads(json_data)
                    generated_text = data["generated_text"]
        if generated_text is not None:
            generated_text = generated_text.strip()
        print(f"Generated text:{generated_text}")
        return generated_text

    # install docker
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.models = [
            {"model_id": "LnL-AI/opt-125M-autoround-lm_head-false-symTrue", "port": 8080},
            {"model_id": "ModelCloud/Meta-Llama-3.1-8B-gptq-4bit", "port": 8081}
        ]

        cls.volume = f"{cls.tmp_dir.name}/data"
        cls.docker_image = "10.0.13.31:5000/huggingface/text-generation-inference:2.2.0"
        cls.docker_process = subprocess.Popen(
            [
                "docker", "run", "--gpus", "all", "--shm-size", "1g",
                "-p", f"{cls.models[0]['port']}:80", "-v", f"{cls.volume}:/data",
                cls.docker_image, "--model-id", cls.models[0]['model_id'], "--quantize", "gptq"
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        cls.docker_process2 = subprocess.Popen(
            [
                "docker", "run", "--gpus", "all", "--shm-size", "1g",
                "-p", f"{cls.models[1]['port']}:80", "-v", f"{cls.volume}:/data",
                cls.docker_image, "--model-id", cls.models[1]['model_id'], "--quantize", "gptq"
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    @classmethod
    def tearDownClass(cls):
        try:
            subprocess.run("docker stop $(docker ps -q)", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop Docker containers: {e}")

        cls.docker_process.terminate()
        cls.docker_process.wait()
        cls.docker_process2.terminate()
        cls.docker_process2.wait()
        cls.tmp_dir.cleanup()
        assert not os.path.exists(cls.tmp_dir.name)

    def test_inference(self):
        start_time = time.time()
        while True:
            for output in iter(self.docker_process.stdout.readline, ''):
                output = output.decode('utf-8')
                print(output)
                if output and "Connected" in output:
                    generated_text = self.generate_request(self.models[0]['port'])
                    self.assertEqual(generated_text,
                                     "Deep learning is a new technology that uses machine learning to learn. It is a new technology")
                    return
            # waitting 120s
            if time.time() - start_time > 120:
                raise TimeoutError("Docker container did not start in the expected time.")

    def test_llama_inference(self):
        start_time = time.time()
        while True:
            for output in iter(self.docker_process2.stdout.readline, ''):
                output = output.decode('utf-8')
                print(output)
                if output and "Connected" in output:
                    generated_text = self.generate_request(self.models[1]['port'])
                    self.assertEqual(generated_text,
                                     "Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning")
                    return
            # waitting 120s
            if time.time() - start_time > 120:
                raise TimeoutError("Docker container did not start in the expected time.")
