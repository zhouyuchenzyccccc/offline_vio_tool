from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Iterable


@dataclass
class ImuSample:
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


_REQUIRED_COLUMNS = {
    "ref_ts_us",
    "accel_ts_us",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_ts_us",
    "gyro_x",
    "gyro_y",
    "gyro_z",
}


def _row_to_timestamp_sec(row: dict[str, str]) -> float:
    ref = row.get("ref_ts_us", "")
    if ref:
        return float(ref) * 1e-6

    at = float(row["accel_ts_us"]) if row.get("accel_ts_us") else 0.0
    gt = float(row["gyro_ts_us"]) if row.get("gyro_ts_us") else 0.0
    if at > 0 and gt > 0:
        return 0.5 * (at + gt) * 1e-6
    if at > 0:
        return at * 1e-6
    return gt * 1e-6


def load_imu_csv(csv_path: str) -> list[ImuSample]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = _REQUIRED_COLUMNS.difference(cols)
        if missing:
            raise ValueError(f"imu.csv missing required columns: {sorted(missing)}")

        out: list[ImuSample] = []
        for row in reader:
            ts = _row_to_timestamp_sec(row)
            out.append(
                ImuSample(
                    timestamp=ts,
                    ax=float(row["accel_x"]),
                    ay=float(row["accel_y"]),
                    az=float(row["accel_z"]),
                    gx=float(row["gyro_x"]),
                    gy=float(row["gyro_y"]),
                    gz=float(row["gyro_z"]),
                )
            )

    out.sort(key=lambda x: x.timestamp)
    return out


def save_orb_imu_txt(imu_samples: Iterable[ImuSample], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for s in imu_samples:
            f.write(
                f"{s.timestamp:.9f} {s.ax:.9f} {s.ay:.9f} {s.az:.9f} {s.gx:.9f} {s.gy:.9f} {s.gz:.9f}\n"
            )
