# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    ndev = torch.cuda.device_count()
    if ndev < 2:
        print(f"Only {ndev} CUDA device(s) visible, need >= 2")
        return

    print("Devices:")
    for d in range(ndev):
        props = torch.cuda.get_device_properties(d)
        print(f"  cuda:{d} {props.name} {props.total_memory/1024**3:.1f} GiB CC {props.major}.{props.minor}")

    print("\nP2P capability (rows=src, cols=dst):")
    for i in range(ndev):
        row = []
        for j in range(ndev):
            if i == j:
                row.append("  - ")
                continue
            row.append("yes" if torch.cuda.can_device_access_peer(i, j) else " no")
        print(f"{i:>2}: " + " ".join(f"{r:>3}" for r in row))

if __name__ == "__main__":
    main()
