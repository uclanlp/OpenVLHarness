# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Color utilities for visualization.
Merges functionality from color_map.py and som_utils.py.
"""

import colorsys
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import numpy as np
import pycocotools.mask as mask_utils


__all__ = [
    "colormap",
    "random_color",
    "random_colors",
    "Color",
    "ColorPalette",
    "DEFAULT_COLOR_HEX_TO_NAME",
    "DEFAULT_COLOR_PALETTE",
    "rgb_to_hex",
    "draw_box",
    "draw_text",
    "draw_mask",
]


# A list of 20 bright and sharp colors for segmentation masks,
# generated from the edges of the sRGB color space for maximum intensity.
_COLORS = (
    np.array(
        [
            # The original 8 sharp colors
            1.000, 1.000, 0.000,  # 1. Yellow
            0.000, 1.000, 0.000,  # 2. Lime
            0.000, 1.000, 1.000,  # 3. Cyan
            1.000, 0.000, 1.000,  # 4. Magenta
            1.000, 0.000, 0.000,  # 5. Red
            1.000, 0.498, 0.000,  # 6. Orange
            0.498, 1.000, 0.000,  # 7. Chartreuse
            0.000, 1.000, 0.498,  # 8. Spring Green
            1.000, 0.000, 0.498,  # 9. Rose
            0.498, 0.000, 1.000,  # 10. Violet
            0.753, 1.000, 0.000,  # 11. Electric Lime
            1.000, 0.753, 0.000,  # 12. Vivid Orange
            0.000, 1.000, 0.753,  # 13. Turquoise
            0.753, 0.000, 1.000,  # 14. Bright Violet
            1.000, 0.000, 0.753,  # 15. Bright Pink
            1.000, 0.251, 0.000,  # 16. Fiery Orange
            0.251, 1.000, 0.000,  # 17. Bright Chartreuse
            0.000, 1.000, 0.251,  # 18. Malachite Green
            0.251, 0.000, 1.000,  # 19. Deep Violet
            1.000, 0.000, 0.251,  # 20. Hot Pink
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def colormap(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a float32 array of Nx3 colors, in range [0, 255] or [0, 1]
    """
    assert maximum in [255, 1], maximum
    c = _COLORS * maximum
    if not rgb:
        c = c[:, ::-1]
    return c


def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def random_colors(N, rgb=False, maximum=255):
    """
    Args:
        N (int): number of unique colors needed
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a list of random_color
    """
    indices = random.sample(range(len(_COLORS)), N)
    ret = [_COLORS[i] * maximum for i in indices]
    if not rgb:
        ret = [x[::-1] for x in ret]
    return ret


def rgb_to_hex(rgb_color):
    """
    Convert a rgb color to hex color.

    Args:
        rgb_color (tuple/list of ints): RGB color in tuple or list format.

    Returns:
        str: Hex color.

    Example:
        >>> rgb_to_hex((255, 0, 244))
        '#ff00f4'
    """
    return "#" + "".join([hex(c)[2:].zfill(2) for c in rgb_color])


DEFAULT_COLOR_HEX_TO_NAME = {
    rgb_to_hex((255, 255, 0)): "yellow",
    rgb_to_hex((0, 255, 0)): "lime",
    rgb_to_hex((0, 255, 255)): "cyan",
    rgb_to_hex((255, 0, 255)): "magenta",
    rgb_to_hex((255, 0, 0)): "red",
    rgb_to_hex((255, 127, 0)): "orange",
    rgb_to_hex((127, 255, 0)): "chartreuse",
    rgb_to_hex((0, 255, 127)): "spring green",
    rgb_to_hex((255, 0, 127)): "rose",
    rgb_to_hex((127, 0, 255)): "violet",
    rgb_to_hex((192, 255, 0)): "electric lime",
    rgb_to_hex((255, 192, 0)): "vivid orange",
    rgb_to_hex((0, 255, 192)): "turquoise",
    rgb_to_hex((192, 0, 255)): "bright violet",
    rgb_to_hex((255, 0, 192)): "bright pink",
    rgb_to_hex((255, 64, 0)): "fiery orange",
    rgb_to_hex((64, 255, 0)): "bright chartreuse",
    rgb_to_hex((0, 255, 64)): "malachite",
    rgb_to_hex((64, 0, 255)): "deep violet",
    rgb_to_hex((255, 0, 64)): "hot pink",
}

DEFAULT_COLOR_PALETTE = list(DEFAULT_COLOR_HEX_TO_NAME.keys())


def _validate_color_hex(color_hex: str):
    color_hex = color_hex.lstrip("#")
    if not all(c in "0123456789abcdefABCDEF" for c in color_hex):
        raise ValueError("Invalid characters in color hash")
    if len(color_hex) not in (3, 6):
        raise ValueError("Invalid length of color hash")


@dataclass
class Color:
    """
    Represents a color in RGB format.

    Attributes:
        r (int): Red channel.
        g (int): Green channel.
        b (int): Blue channel.
    """

    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, color_hex: str):
        """
        Create a Color instance from a hex string.

        Args:
            color_hex (str): Hex string of the color.

        Returns:
            Color: Instance representing the color.
        """
        _validate_color_hex(color_hex)
        color_hex = color_hex.lstrip("#")
        if len(color_hex) == 3:
            color_hex = "".join(c * 2 for c in color_hex)
        r, g, b = (int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))
        return cls(r, g, b)

    @classmethod
    def to_hex(cls, color):
        """
        Convert a Color instance to a hex string.
        """
        return rgb_to_hex((color.r, color.g, color.b))

    def as_rgb(self) -> Tuple[int, int, int]:
        """Returns the color as an RGB tuple."""
        return self.r, self.g, self.b

    def as_bgr(self) -> Tuple[int, int, int]:
        """Returns the color as a BGR tuple."""
        return self.b, self.g, self.r

    @classmethod
    def white(cls):
        return Color.from_hex(color_hex="#ffffff")

    @classmethod
    def black(cls):
        return Color.from_hex(color_hex="#000000")

    @classmethod
    def red(cls):
        return Color.from_hex(color_hex="#ff0000")

    @classmethod
    def green(cls):
        return Color.from_hex(color_hex="#00ff00")

    @classmethod
    def blue(cls):
        return Color.from_hex(color_hex="#0000ff")


@dataclass
class ColorPalette:
    colors: List[Color]

    @classmethod
    def default(cls):
        """Returns a default color palette."""
        return ColorPalette.from_hex(color_hex_list=DEFAULT_COLOR_PALETTE)

    @classmethod
    def from_hex(cls, color_hex_list: List[str]):
        """Create a ColorPalette instance from a list of hex strings."""
        colors = [Color.from_hex(color_hex) for color_hex in color_hex_list]
        return cls(colors)

    def by_idx(self, idx: int) -> Color:
        """Return the color at a given index in the palette."""
        if idx < 0:
            raise ValueError("idx argument should not be negative")
        idx = idx % len(self.colors)
        return self.colors[idx]

    def find_farthest_color(self, img_array):
        """
        Return the color that is the farthest from the given color.

        Args:
            img_array (np array): any *x3 np array, 3 is the RGB color channel.

        Returns:
            Color: Farthest color.
        """
        img_array = img_array.reshape((-1, 3))
        color_values = np.array([[c.r, c.g, c.b] for c in self.colors])
        distances = np.sqrt(
            np.sum((img_array[:, np.newaxis, :] - color_values) ** 2, axis=2)
        )
        mean_distances = np.mean(distances, axis=0)
        farthest_idx = np.argmax(mean_distances)
        farthest_color = self.colors[farthest_idx]
        farthest_color_hex = Color.to_hex(farthest_color)
        if farthest_color_hex in DEFAULT_COLOR_HEX_TO_NAME:
            farthest_color_name = DEFAULT_COLOR_HEX_TO_NAME[farthest_color_hex]
        else:
            farthest_color_name = "unknown"

        return farthest_color, farthest_color_name


def _change_color_brightness(color, brightness_factor):
    """
    Depending on the brightness_factor, gives a lighter or darker color.

    Args:
        color: color of the polygon. Refer to `matplotlib.colors` for formats.
        brightness_factor (float): a value in [-1.0, 1.0] range.

    Returns:
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
    """
    assert brightness_factor >= -1.0 and brightness_factor <= 1.0
    color = mplc.to_rgb(color)
    polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
    modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
    modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
    modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
    modified_color = colorsys.hls_to_rgb(
        polygon_color[0], modified_lightness, polygon_color[2]
    )
    return modified_color


def draw_box(ax, box_coord, alpha=0.8, edge_color="g", line_style="-", linewidth=2.0):
    """Draw a bounding box on a matplotlib axis."""
    x0, y0, width, height = box_coord
    ax.add_patch(
        mpl.patches.Rectangle(
            (x0, y0),
            width,
            height,
            fill=False,
            edgecolor=edge_color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=line_style,
        )
    )


def draw_text(
    ax,
    text,
    position,
    font_size=None,
    color="g",
    horizontal_alignment="left",
    rotation=0,
):
    """Draw text on a matplotlib axis."""
    if not font_size:
        font_size = mpl.rcParams["font.size"]

    color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    color[np.argmax(color)] = max(0.8, np.max(color))

    x, y = position
    ax.text(
        x,
        y,
        text,
        size=font_size,
        family="sans-serif",
        bbox={"facecolor": "none", "alpha": 0.5, "pad": 0.7, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        rotation=rotation,
    )


def draw_mask(
    ax, rle, color, show_holes=True, alpha=0.15, upsample_factor=1.0, rle_upsampled=None
):
    """Draw a mask on a matplotlib axis."""
    if isinstance(rle, dict):
        mask = mask_utils.decode(rle)
    elif isinstance(rle, np.ndarray):
        mask = rle
    else:
        raise ValueError(f"Unsupported type for rle: {type(rle)}")

    mask_upsampled = None
    if upsample_factor > 1.0 and show_holes:
        assert rle_upsampled is not None
        if isinstance(rle_upsampled, dict):
            mask_upsampled = mask_utils.decode(rle_upsampled)
        elif isinstance(rle_upsampled, np.ndarray):
            mask_upsampled = rle_upsampled
        else:
            raise ValueError(f"Unsupported type for rle: {type(rle)}")

    if show_holes:
        if mask_upsampled is None:
            mask_upsampled = mask
        h, w = mask_upsampled.shape
        mask_img = np.zeros((h, w, 4))
        mask_img[:, :, :-1] = color[np.newaxis, np.newaxis, :]
        mask_img[:, :, -1] = mask_upsampled * alpha
        ax.imshow(mask_img)

    *_, contours, _ = cv2.findContours(
        mask.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    upsampled_contours = [(cont + 0.5) * upsample_factor - 0.5 for cont in contours]
    facecolor = (0, 0, 0, 0) if show_holes else color
    if alpha > 0.8:
        edge_color = _change_color_brightness(color, brightness_factor=-0.7)
    else:
        edge_color = color
    for cont in upsampled_contours:
        polygon = mpl.patches.Polygon(
            [el[0] for el in cont],
            edgecolor=edge_color,
            linewidth=2.0,
            facecolor=facecolor,
        )
        ax.add_patch(polygon)
