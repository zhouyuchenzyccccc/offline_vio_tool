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

_RAW_COLUMNS = {"ts_us", "sensor", "x", "y", "z"}


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
        out: list[ImuSample] = []

        # Format A (legacy): one row already contains accel+gyro fields.
        if _REQUIRED_COLUMNS.issubset(cols):
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

        # Format B (raw): ts_us,sensor,x,y,z with separate accel/gyro rows.
        elif _RAW_COLUMNS.issubset(cols):
            accel_rows: list[tuple[float, float, float, float]] = []
            gyro_rows: list[tuple[float, float, float, float]] = []

            for row in reader:
                sensor = (row.get("sensor") or "").strip().lower()
                ts = float(row["ts_us"]) * 1e-6
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])

                if sensor in {"accel", "accelerometer", "acc"}:
                    accel_rows.append((ts, x, y, z))
                elif sensor in {"gyro", "gyroscope", "gyr"}:
                    gyro_rows.append((ts, x, y, z))

            accel_rows.sort(key=lambda r: r[0])
            gyro_rows.sort(key=lambda r: r[0])

            # Pair accel and gyro streams by nearest timestamp.
            i = 0
            j = 0
            max_pair_dt_sec = 0.005  # 5 ms tolerance for sensor timestamp skew
            while i < len(accel_rows) and j < len(gyro_rows):
                ta, ax, ay, az = accel_rows[i]
                tg, gx, gy, gz = gyro_rows[j]
                dt = ta - tg

                if abs(dt) <= max_pair_dt_sec:
                    out.append(
                        ImuSample(
                            timestamp=0.5 * (ta + tg),
                            ax=ax,
                            ay=ay,
                            az=az,
                            gx=gx,
                            gy=gy,
                            gz=gz,
                        )
                    )
                    i += 1
                    j += 1
                elif dt < 0:
                    i += 1
                else:
                    j += 1

        else:
            missing_legacy = sorted(_REQUIRED_COLUMNS.difference(cols))
            missing_raw = sorted(_RAW_COLUMNS.difference(cols))
            raise ValueError(
                "Unsupported IMU csv schema. "
                f"Legacy missing columns: {missing_legacy}; "
                f"Raw missing columns: {missing_raw}."
            )

    out.sort(key=lambda x: x.timestamp)
    return out


def save_orb_imu_txt(imu_samples: Iterable[ImuSample], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for s in imu_samples:
            f.write(
                f"{s.timestamp:.9f} {s.ax:.9f} {s.ay:.9f} {s.az:.9f} {s.gx:.9f} {s.gy:.9f} {s.gz:.9f}\n"
            )
