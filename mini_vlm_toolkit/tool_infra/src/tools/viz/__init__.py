# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Visualization utilities for the grounding agent.
"""

from .core import visualize
from .visualizer import Visualizer, VisImage, GenericMask, ColorMode
from .zoom_in import render_zoom_in
from .colors import (
    Color,
    ColorPalette,
    colormap,
    random_color,
    random_colors,
    rgb_to_hex,
    draw_box,
    draw_text,
    draw_mask,
)
from .boxes import Boxes, BoxMode
from .masks import BitMasks, PolygonMasks, ROIMasks, rle_encode, robust_rle_encode

__all__ = [
    # Core
    "visualize",
    # Visualizer
    "Visualizer",
    "VisImage",
    "GenericMask",
    "ColorMode",
    # Zoom
    "render_zoom_in",
    # Colors
    "Color",
    "ColorPalette",
    "colormap",
    "random_color",
    "random_colors",
    "rgb_to_hex",
    "draw_box",
    "draw_text",
    "draw_mask",
    # Boxes
    "Boxes",
    "BoxMode",
    # Masks
    "BitMasks",
    "PolygonMasks",
    "ROIMasks",
    "rle_encode",
    "robust_rle_encode",
]
