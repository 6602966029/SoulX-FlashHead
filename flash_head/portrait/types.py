from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class PortraitOptions:
    output_size: tuple = (1080, 1920)
    fps: int = 25
    canvas_margin: int = 64


@dataclass(frozen=True)
class PortraitContext:
    canvas: np.ndarray
    head_rgb: np.ndarray
    head_path: Path
    head_to_canvas: np.ndarray
    landmarks: np.ndarray
    face_width: float
    options: PortraitOptions


@dataclass(frozen=True)
class GenerationResult:
    path: str
    frame_count: int
    held_frames: int
