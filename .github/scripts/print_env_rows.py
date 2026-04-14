import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-envs-json", default="[]")
    parser.add_argument("--model-envs-json", default="[]")
    args = parser.parse_args()

    torch_envs = json.loads(args.torch_envs_json or "[]")
    model_envs = json.loads(args.model_envs_json or "[]")

    for item in [*torch_envs, *model_envs]:
        print(f'{item["env_name"]}\t{item["python_version"]}')


if __name__ == "__main__":
    main()
