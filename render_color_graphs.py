from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ----------------------------
# 2) Rendering utilities
# ----------------------------
@dataclass(frozen=True)
class RenderStyle:
    img_size: int = 128
    line_w: int = 3              # connection line width in pixels
    node_size: int = 3           # node square size (3x3)
    padding: int = 4             # padding around the normalized drawing

    bg: Tuple[int, int, int] = (0, 0, 0)
    line: Tuple[int, int, int] = (255, 255, 255)

    center: Tuple[int, int, int] = (255, 0, 0)     # red
    entrance: Tuple[int, int, int] = (0, 255, 0)   # green
    corridor: Tuple[int, int, int] = (0, 0, 255)   # blue


def _normalize_points_to_image(
    xs: np.ndarray,
    zs: np.ndarray,
    img_size: int,
    padding: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize (x,z) into pixel coordinates [padding .. img_size-1-padding].
    Keeps aspect ratio, centers content.
    """
    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)

    min_x, max_x = float(xs.min()), float(xs.max())
    min_z, max_z = float(zs.min()), float(zs.max())

    # Handle degenerate cases (all points same x or z)
    span_x = max(max_x - min_x, 1e-9)
    span_z = max(max_z - min_z, 1e-9)

    avail = (img_size - 1) - 2 * padding
    scale = avail / max(span_x, span_z)

    # Scale into [0..avail], then shift by padding, then center shorter axis
    x0 = (xs - min_x) * scale
    z0 = (zs - min_z) * scale

    used_w = span_x * scale
    used_h = span_z * scale

    # Centering offsets (within avail area)
    off_x = (avail - used_w) / 2.0
    off_z = (avail - used_h) / 2.0

    xp = x0 + padding + off_x
    zp = z0 + padding + off_z

    # Convert to int pixels
    return np.rint(xp).astype(int), np.rint(zp).astype(int)


def render_file_to_image(
    df: pd.DataFrame,
    file_id,
    style: RenderStyle = RenderStyle(),
) -> Image.Image:
    """
    Render one room (one file_id) to a PIL Image (RGB) of size style.img_size x style.img_size.

    - Uses columns: file_id, node_type, x, z, connections
    - Ignores y
    - Draws white connection lines then node squares on top
    """
    g = df[df["file_id"] == file_id].copy()
    if g.empty:
        raise ValueError(f"No rows found for file_id={file_id!r}")

    # Make sure connections are list-like
    g["connections"] = g["connections"].apply(lambda x: list(x) if isinstance(x, list) else [])

    # Normalize coordinates to pixels
    xs = g["x"].to_numpy(dtype=float)
    zs = g["z"].to_numpy(dtype=float)
    xp, zp = _normalize_points_to_image(xs, zs, style.img_size, style.padding)

    # Store pixel coords back aligned to g's row order
    g = g.reset_index(drop=False).rename(columns={"index": "_global_idx"})
    g["_px"] = xp
    g["_pz"] = zp

    # Build mapping global row index -> local row position -> pixel
    global_to_local = {int(r._global_idx): i for i, r in g.iterrows()}

    img = Image.new("RGB", (style.img_size, style.img_size), style.bg)
    draw = ImageDraw.Draw(img)

    # ---- draw edges (white)
    # Avoid drawing duplicates: store undirected edges as (min,max)
    drawn = set()
    for i, r in g.iterrows():
        a_global = int(r["_global_idx"])
        ax, az = int(r["_px"]), int(r["_pz"])

        for b_global in r["connections"]:
            if b_global not in global_to_local:
                continue  # ignore cross-file or invalid references
            edge = (min(a_global, b_global), max(a_global, b_global))
            if edge in drawn:
                continue
            drawn.add(edge)

            j = global_to_local[b_global]
            bx, bz = int(g.at[j, "_px"]), int(g.at[j, "_pz"])
            draw.line([(ax, az), (bx, bz)], fill=style.line, width=style.line_w)

    # ---- draw nodes on top (3x3 squares)
    half = style.node_size // 2

    def node_color(t: str) -> Tuple[int, int, int]:
        if t == "center":
            return style.center
        if t == "entrance":
            return style.entrance
        if t == "corridor":
            return style.corridor
        # default (if you have other types)
        return (200, 200, 200)

    for _, r in g.iterrows():
        t = str(r["node_type"])
        cx, cz = int(r["_px"]), int(r["_pz"])
        col = node_color(t)

        x0, y0 = cx - half, cz - half
        x1, y1 = cx + half, cz + half

        # clip to bounds
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(style.img_size - 1, x1); y1 = min(style.img_size - 1, y1)

        draw.rectangle([x0, y0, x1, y1], fill=col)

    return img


def save_file_image_png(
    df: pd.DataFrame,
    file_id,
    out_path: str,
    style: RenderStyle = RenderStyle(),
) -> None:
    """
    Convenience wrapper to render and save as PNG.
    """
    img = render_file_to_image(df, file_id=file_id, style=style)
    img.save(out_path, format="PNG")