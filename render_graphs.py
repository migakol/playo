from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Set
import numpy as np
import pandas as pd

# Pillow for drawing
from PIL import Image, ImageDraw

# -----------------------------
# 2) Rendering: 32x32 images from X,Z + connection lines
# -----------------------------
@dataclass(frozen=True)
class RenderParams:
    size: int = 32              # image width/height
    pad: int = 2               # padding in pixels from border
    line_width: int = 1         # line thickness in pixels
    white: int = 255
    black: int = 0


def _scale_points_to_canvas(
    x: np.ndarray,
    z: np.ndarray,
    *,
    size: int,
    pad: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map (x,z) to integer pixel coords in [pad, size-1-pad], preserving aspect ratio.
    """
    x = x.astype(np.float64)
    z = z.astype(np.float64)

    min_x, max_x = float(np.min(x)), float(np.max(x))
    min_z, max_z = float(np.min(z)), float(np.max(z))

    span_x = max(max_x - min_x, 1e-9)
    span_z = max(max_z - min_z, 1e-9)

    # target drawable area
    lo = pad
    hi = size - 1 - pad
    drawable = max(hi - lo, 1)

    # preserve aspect ratio: scale by the larger span
    scale = drawable / max(span_x, span_z)

    # centered within drawable area
    x0 = (x - min_x) * scale
    z0 = (z - min_z) * scale

    used_w = span_x * scale
    used_h = span_z * scale
    off_x = lo + (drawable - used_w) / 2.0
    off_z = lo + (drawable - used_h) / 2.0

    # Note: image y grows downward; we can flip z so "up" in z goes upward visually if you want
    px = np.round(x0 + off_x).astype(int)
    pz = np.round(z0 + off_z).astype(int)

    # clamp
    px = np.clip(px, lo, hi)
    pz = np.clip(pz, lo, hi)
    return px, pz


def render_file_to_image(
    g: pd.DataFrame,
    params: RenderParams = RenderParams(),
) -> np.ndarray:
    """
    Render a single file-group DataFrame (same file_id) into a 32x32 grayscale image.
    Black background, white connection lines.

    Assumes:
    - g.index are the node IDs referenced by 'connections' (global df.index).
    - 'connections' lists contain those indices (within same file).
    """
    if g.empty:
        return np.zeros((params.size, params.size), dtype=np.uint8)

    # Extract coords (ignore y)
    x = g["x"].to_numpy(dtype=np.float64)
    z = g["z"].to_numpy(dtype=np.float64)

    px, pz = _scale_points_to_canvas(x, z, size=params.size, pad=params.pad)

    # Build mapping from node index -> (px, pz)
    idx_list = list(g.index)
    idx_to_pos = {idx: (int(px[i]), int(pz[i])) for i, idx in enumerate(idx_list)}

    # Create image
    img = Image.new("L", (params.size, params.size), color=params.black)
    draw = ImageDraw.Draw(img)

    # Draw edges (undirected; avoid double draw)
    drawn: Set[Tuple[int, int]] = set()

    for i_idx in idx_list:
        conns = g.at[i_idx, "connections"]
        if not conns:
            continue
        for j_idx1 in conns:
            j_idx = idx_list[j_idx1]
            if j_idx not in idx_to_pos:
                continue
            a, b = (i_idx, j_idx) if i_idx <= j_idx else (j_idx, i_idx)
            if a == b or (a, b) in drawn:
                continue
            drawn.add((a, b))
            (x1, y1) = idx_to_pos[a]
            (x2, y2) = idx_to_pos[b]
            draw.line([(x1, y1), (x2, y2)], fill=params.white, width=params.line_width)

    return np.array(img, dtype=np.uint8)


def df_to_32x32_images(
    df_clean: pd.DataFrame,
    params: RenderParams = RenderParams(),
) -> Dict[str, np.ndarray]:
    """
    Convert each file_id into a 32x32 grayscale image array (uint8).
    Returns dict: file_id -> image (H,W).
    """
    images: Dict[str, np.ndarray] = {}
    for file_id, g in df_clean.groupby("file_id", sort=False):
        images[file_id] = render_file_to_image(g, params=params)
    return images

def one_room_df_to_32x32_images(
    df_clean: pd.DataFrame,
    params: RenderParams = RenderParams(),
    room_id: int = 0
) -> Dict[str, np.ndarray]:
    """
    Convert each file_id into a 32x32 grayscale image array (uint8).
    Returns dict: file_id -> image (H,W).
    """
    images: Dict[str, np.ndarray] = {}
    for file_id, g in df_clean.groupby("file_id", sort=False):
        # if file_id != room_id:
        #     continue
        if file_id % 500 == 0:
            print(file_id)
        img_arr = render_file_to_image(g, params=params)

        img = Image.fromarray(img_arr, mode="L")  # L = grayscale
        img.save(f"/Users/michaelkolomenkin/Data/playo/smaller_images/room_{file_id}.png")
        # images[file_id] = render_file_to_image(g, params=params)
    return images