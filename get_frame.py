from __future__ import annotations

import csv
from itertools import islice
from pathlib import Path

import cv2
import numpy as np


DEFAULT_CAMERA_DIR = Path(__file__).resolve().parent / "MH_02_easy" / "mav0" / "cam0"


def get_frame(
    frame_index: int = 0,
    camera_dir: str | Path = DEFAULT_CAMERA_DIR,
    grayscale: bool = True,
) -> np.ndarray:
    if frame_index < 0:
        raise IndexError("frame_index must be non-negative")

    camera_dir = Path(camera_dir)
    with (camera_dir / "data.csv").open(encoding="utf-8", newline="") as csv_file:
        rows = csv.reader(
            line for line in csv_file if line.strip() and not line.startswith("#")
        )
        row = next(islice(rows, frame_index, None), None)

    if row is None:
        raise IndexError(f"Frame index out of range: {frame_index}")

    image_path = camera_dir / "data" / row[1].strip()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    frame = cv2.imread(str(image_path), flag)
    if frame is None:
        raise OSError(f"OpenCV could not read the image: {image_path}")

    return frame
