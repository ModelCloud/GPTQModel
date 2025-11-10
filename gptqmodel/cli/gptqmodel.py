import argparse
from typing import Iterable

from gptqmodel.cli.env import _handle_env_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gptqmodel", description="Utilities for GPTQModel")
    subparsers = parser.add_subparsers(dest="command")

    env_parser = subparsers.add_parser("env", help="Inspect the local GPTQModel runtime environment")
    env_parser.set_defaults(func=_handle_env_command)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - manual execution entrypoint
    main()
