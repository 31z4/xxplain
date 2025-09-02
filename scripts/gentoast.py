#!/usr/bin/env python3
import csv
import os
import sys
from datetime import datetime, timezone


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def format_timestamp_utc_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S.%f+00")


def generate_random_hex_bytes(size_bytes: int) -> str:
    return "\\x" + os.urandom(size_bytes).hex()


def write_toast_csv(
    output_dir: str,
    filename: str = "toast.csv",
    num_rows: int = 1_000,
    data_size_bytes: int = 1_024 * 1_024,  # 1 MB
) -> None:
    ensure_directory(output_dir)

    output_path = os.path.join(output_dir, filename)

    # Allow very large CSV fields (2MB+ per row for hex data)
    try:
        csv.field_size_limit(sys.maxsize)
    except (OverflowError, ValueError):
        # Fallback to a large, but safe value
        csv.field_size_limit(2_147_483_647)

    # Use a large buffer to reduce syscall overhead
    with open(output_path, "w", newline="", buffering=1024 * 1024) as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(["id", "data", "created_at", "updated_at"])

        for row_id in range(1, num_rows + 1):
            timestamp = format_timestamp_utc_now()
            data_hex = generate_random_hex_bytes(data_size_bytes)
            writer.writerow([row_id, data_hex, timestamp, timestamp])


def main() -> None:
    output_dir = os.path.join("data", "toast")
    write_toast_csv(output_dir)


if __name__ == "__main__":
    main()
