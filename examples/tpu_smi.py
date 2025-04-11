import time
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
from tabulate import tabulate

def get_memory_table():
    mem_info = xm.get_memory_info(xm.xla_device())
    print(f"{mem_info}")
    try:
        used = int(mem_info['bytes_used'])
        limit = int(mem_info['bytes_limit'])
        peak = int(mem_info['peak_bytes_used'])

        data = [
            ["Used (MB)", used // (1024 * 1024)],
            ["Peak (MB)", peak // (1024 * 1024)],
            ["Limit (MB)", limit // (1024 * 1024)],
        ]
        return data
    except Exception as e:
        return [["Error", f"Failed to parse memory info: {e}"]]

def get_metrics_summary():
    report = met.metrics_report().split('\n')
    summary = []
    for line in report:
        if ':' in line and any(key in line for key in ["Time", "Rate", "Size", "Count"]):
            parts = line.strip().split(':')
            summary.append([parts[0].strip(), parts[1].strip()])
    return summary

def monitor_tpu(interval=1):
    print("🧠 Starting TPU Monitor (updates every {}s)\n".format(interval))
    while True:
        mem_table = get_memory_table()
        metrics_table = get_metrics_summary()

        print("\n=== TPU Memory Usage ===")
        print(tabulate(mem_table, headers=["Metric", "Value"], tablefmt="grid"))

        print("\n=== TPU Execution Metrics ===")
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="grid"))

        time.sleep(interval)

if __name__ == "__main__":
    monitor_tpu(interval=1)
