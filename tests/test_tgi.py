# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile # noqa: E402
import unittest # noqa: E402
import subprocess # noqa: E402
import time # noqa: E402
import requests # noqa: E402
import json # noqa: E402

class TestModelInference(unittest.TestCase):
    # install docker
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        cls.volume = f"{cls.tmp_dir.name}/data"
        cls.docker_image = "ghcr.io/huggingface/text-generation-inference:2.2.0"
        cls.port = 8080

        cls.docker_process = subprocess.Popen(
            [
                "docker", "run", "--gpus", "all", "--shm-size", "1g",
                "-p", f"{cls.port}:80", "-v", f"{cls.volume}:/data",
                cls.docker_image, "--model-id", cls.MODEL_ID
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        start_time = time.time()
        while True:
            for output in iter(cls.docker_process.stdout.readline, ''):
                output = output.decode('utf-8')
                print(output)
                if output:
                    print(output.strip())  # 打印 stdout
                    if "Connected" in output:
                        print("Docker container is ready.")
                        return
            # waitting 120s
            if time.time() - start_time > 120:
                raise TimeoutError("Docker container did not start in the expected time.")

            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        try:
            subprocess.run("docker stop $(docker ps -q)", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to stop Docker containers: {e}")

        cls.docker_process.terminate()
        cls.docker_process.wait()
        cls.tmp_dir.cleanup()
        assert not os.path.exists(cls.tmp_dir.name)

    def test_inference(self):
            url = f"http://127.0.0.1:{self.port}/generate_stream"
            headers = {'Content-Type': 'application/json'}
            payload = {
                "inputs": "What is Deep Learning?",
                "parameters": {"max_new_tokens": 20}
            }
            response = requests.post(url, json=payload, headers=headers, stream=True)
            self.assertEqual(response.status_code, 200)
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
            self.assertEqual(generated_text, "Deep Learning is a type of machine learning that involves the use of deep neural networks.")
