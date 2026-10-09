# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""
Visualizer class for drawing detection/segmentation results on images.
Trimmed version focused on overlay_instances functionality.
"""

import colorsys
import logging
from enum import Enum, unique

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .boxes import Boxes, BoxMode
from .colors import random_color
from .masks import BitMasks, PolygonMasks

logger = logging.getLogger(__name__)


__all__ = ["ColorMode", "VisImage", "Visualizer", "GenericMask"]


_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        """Create a mask wrapper from polygons, COCO RLE or a binary array.

        Args:
            mask_or_polygons: RLE dictionary, list of flattened polygons, or H x W array.
            height: Target image height in pixels.
            width: Target image width in pixels.

        Raises:
            ValueError: The mask representation is unsupported.
            AssertionError: Array or RLE dimensions do not match the image.
        """
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        m = mask_or_polygons
        if isinstance(m, dict):
            # RLEs
            assert "counts" in m and "size" in m
            if isinstance(m["counts"], list):  # uncompressed RLEs
                h, w = m["size"]
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return

        if isinstance(m, list):  # list[ndarray]
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return

        if isinstance(m, np.ndarray):  # assumed to be a binary mask
            assert m.shape[1] != 2, m.shape
            assert m.shape == (
                height,
                width,
            ), f"mask shape: {m.shape}, target dims: {height}, {width}"
            self._mask = m.astype("uint8")
            return

        raise ValueError(
            "GenericMask cannot handle object {} of type '{}'".format(m, type(m))
        )

    @property
    def mask(self):
        """Return the binary bitmap, converting and caching polygon data if necessary."""
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        """Return flattened polygon coordinates, extracting and caching contours when needed."""
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        """Return whether bitmap contours contain holes; polygon inputs are treated as hole-free."""
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = (
                    False  # if original format is polygon, does not have holes
                )
        return self._has_holes

    def mask_to_polygons(self, mask):
        """Return pixel-centered contour coordinates and whether the binary mask has holes."""
        mask = np.ascontiguousarray(mask)
        res = cv2.findContours(
            mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes

    def polygons_to_mask(self, polygons):
        """Rasterize polygons into a binary array with the configured image dimensions."""
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        """Return the number of foreground pixels."""
        return self.mask.sum()

    def bbox(self):
        """Return the mask's bounding box in absolute xyxy coordinates."""
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox


class VisImage:
    """A matplotlib-backed RGB image canvas for drawing and saving overlays."""
    def __init__(self, img, scale=1.0):
        """Create a canvas for an RGB image at the requested display scale.

        Args:
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """Create an axis-free canvas sized to the source image and configured scale."""
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """Display a uint8 source image on the canvas using image pixel coordinates."""
        img = img.astype("uint8")
        self.ax.imshow(
            img, extent=(0, self.width, self.height, 0), interpolation="nearest"
        )

    def save(self, filepath):
        """Save the rendered canvas to a local file using matplotlib format inference."""
        self.fig.savefig(filepath)

    def get_image(self):
        """Return the rendered canvas as a uint8 RGB array.

        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer:
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}`
    that draw primitive objects to images, as well as high-level wrappers like
    `overlay_instances` that draw composite data in some pre-defined style.
    """

    def __init__(
        self,
        img_rgb,
        metadata=None,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
        font_size_multiplier=1.3,
        border_width_multiplier=1.5,
    ):
        """Initialize image visualization settings and the output canvas.

        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.border_width_multiplier = border_width_multiplier
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = (
            max(np.sqrt(self.output.height * self.output.width) // 60, 15 // scale)
            * font_size_multiplier
        )
        self._instance_mode = instance_mode

        import matplotlib.colors as mcolors

        css4_colors = mcolors.CSS4_COLORS
        self.color_proposals = [
            list(mcolors.hex2color(color)) for color in css4_colors.values()
        ]

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        binary_masks=None,
        alpha=0.5,
        label_mode="1",
    ):
        """Draw instance boxes, masks and labels on the output canvas.

        Args:
            boxes (Boxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image.
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image.
        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(num_instances)
            ]
        if num_instances == 0:
            return labels, [], []
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        marks = []
        marks_position = []
        added_positions = set()
        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], alpha=1, edge_color=color)
                if binary_masks is None:
                    # draw number for non-mask instances
                    mark = self._draw_number_in_box(
                        boxes[i], i + 1, color=color, label_mode=label_mode
                    )
                    marks.append(mark)

            if binary_masks is not None:
                mark, mask_position = self._draw_number_in_mask(
                    binary_mask=binary_masks[i].astype("uint8"),
                    text=i + 1,
                    color=color,
                    added_positions=added_positions,
                    label_mode=label_mode,
                )
                marks.append(mark)
                marks_position.append(mask_position)

                self.draw_binary_mask(
                    binary_masks[i],
                    color=color,
                    edge_color=_OFF_WHITE,
                    alpha=alpha,
                )

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(
                        segment.reshape(-1, 2), color, alpha=0
                    )  # alpha=0 so holes in masks are not colored

        return labels, marks, marks_position

    def overlay_rotated_instances(self, boxes=None, labels=None, assigned_colors=None):
        """Draw rotated instance boxes and their labels.

        Args:
            boxes (ndarray): an Nx5 numpy array of
                (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image.
            labels (list[str]): the text to be displayed for each instance.
            assigned_colors (list[matplotlib.colors]): a list of colors.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = len(boxes)

        if assigned_colors is None:
            assigned_colors = [
                random_color(rgb=True, maximum=1) for _ in range(num_instances)
            ]
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        if boxes is not None:
            areas = boxes[:, 2] * boxes[:, 3]

        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            self.draw_rotated_box_with_label(
                boxes[i],
                edge_color=colors[i],
                label=labels[i] if labels is not None else None,
            )

        return self.output

    def mask_dims_from_binary(self, binary_mask):
        """Return foreground x/y spans for a nonempty binary mask."""
        ind_y, ind_x = np.where(binary_mask == 1)
        min_ind_x = np.min(ind_x)
        max_ind_x = np.max(ind_x)
        min_ind_y = np.min(ind_y)
        max_ind_y = np.max(ind_y)
        return (max_ind_x - min_ind_x), (max_ind_y - min_ind_y)

    def reposition_label(self, position, cur, binary_mask, move_count):
        """Propose a label displacement to separate small masks or nearby labels.

        Returns (should_move, delta_x, delta_y), respecting configured image-edge
        margins for collision adjustments.
        """
        img_width, img_height = self.output.width, self.output.height
        mask_width, mask_height = self.mask_dims_from_binary(binary_mask)

        # set resposition thresholds
        mask_width_limit, mask_height_limit = (25, 25)
        location_diff_threshold = 15
        x_boundry_limit, y_boundry_limit = (20, 20)

        offset_x = 15
        offset_y = 15

        x1, y1 = position

        if (
            mask_width < mask_width_limit
            and mask_height < mask_height_limit
            and move_count == 0
        ):
            move_x = offset_x if offset_x + x1 < img_width else -offset_x
            move_y = offset_y if offset_y + y1 < img_height else -offset_y
            return (True, move_x, move_y)

        for x2, y2 in cur:
            if abs(x1 - x2) + abs(y1 - y2) < location_diff_threshold:
                move_x = offset_x if x1 >= x2 else -offset_x
                move_y = offset_y if y1 >= y2 else -offset_y
                move_x = (
                    0
                    if x1 + move_x > img_width - x_boundry_limit
                    or x1 + move_x < x_boundry_limit
                    else move_x
                )
                move_y = (
                    0
                    if y1 + move_y > img_height - y_boundry_limit
                    or y1 + move_y < y_boundry_limit
                    else move_y
                )
                return (True, move_x, move_y)
        return (False, 0, 0)

    def locate_label_position(self, original_position, added_positions, binary_mask):
        """Find and record a label location using at most ten collision adjustments.

        Returns the requested position when positions or mask data are unavailable;
        otherwise adds the final position to added_positions.
        """
        if added_positions is None or binary_mask is None:
            return original_position

        x, y = original_position

        move_count = 0
        reposition, x_move, y_move = self.reposition_label(
            (x, y), added_positions, binary_mask, move_count
        )
        while reposition and move_count < 10:
            x += x_move
            y += y_move
            move_count += 1
            reposition, x_move, y_move = self.reposition_label(
                (x, y), added_positions, binary_mask, move_count
            )
        added_positions.add((x, y))
        return x, y

    # ----------------------------------------------------------------
    # Primitive drawing functions
    # ----------------------------------------------------------------

    def draw_text(
        self,
        text,
        position,
        added_positions=None,
        binary_mask=None,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """Draw text at a pixel location using the requested alignment and styling.

        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text.
            color: color of the text.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.15)
        color[np.argmax(color)] = max(0.8, np.max(color))

        def contrasting_color(rgb):
            R, G, B = rgb
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            return "black" if Y > 128 else "white"

        bbox_background = contrasting_color(color * 255)

        x, y = self.locate_label_position(
            original_position=position,
            added_positions=added_positions,
            binary_mask=binary_mask,
        )

        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={
                "facecolor": bbox_background,
                "alpha": 0.8,
                "pad": 0.7,
                "edgecolor": "none",
            },
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """Draw an axis-aligned box on the output canvas.

        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates.
            alpha (float): blending efficient.
            edge_color: color of the outline of the box.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 12, 1) * self.border_width_multiplier

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def draw_rotated_box_with_label(
        self, rotated_box, alpha=0.5, edge_color="g", line_style="-", label=None
    ):
        """
        Draw a rotated box with label on its top-left corner.
        """
        import math

        cnt_x, cnt_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        rotated_rect = [
            (s * yy + c * xx + cnt_x, c * yy - s * xx + cnt_y) for (xx, yy) in rect
        ]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # topleft corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(
                edge_color, brightness_factor=0.7
            )
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                * 0.5
                * self._default_font_size
            )
            self.draw_text(
                label, text_pos, color=label_color, font_size=font_size, rotation=angle
            )

        return self.output

    def draw_circle(self, circle_coord, color, radius=3):
        """Draw a circle at an image-coordinate location.

        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        """Draw connected x/y coordinates on the output canvas.

        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
            y_data (list[int]): a list containing y values of all the points being drawn.
            color: color of the line.
            linestyle: style of the line.
            linewidth (float or None): width of the line.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_binary_mask(
        self,
        binary_mask,
        color=None,
        *,
        edge_color=None,
        text=None,
        alpha=0.7,
        area_threshold=10,
    ):
        """Overlay a binary mask with optional edges and a text label.

        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask.
            edge_color: color of the polygon edges.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient.
            area_threshold (float): a connected component smaller than this area will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)

        has_valid_segment = False
        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        if not mask.has_holes:
            # draw polygons for regular masks
            for segment in mask.polygons:
                area = mask_util.area(
                    mask_util.frPyObjects([segment], shape2d[0], shape2d[1])
                )
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(
                    segment, color=color, edge_color=edge_color, alpha=alpha
                )
        else:
            # https://stackoverflow.com/questions/8919719/how-to-plot-a-complex-polygon
            rgba = np.zeros(shape2d + (4,), dtype="float32")
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype("float32") * alpha
            has_valid_segment = True
            self.output.ax.imshow(
                rgba, extent=(0, self.output.width, self.output.height, 0)
            )

        if text is not None and has_valid_segment:
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            self._draw_text_in_mask(binary_mask, text, lighter_color)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        """Draw a polygon with a filled face and optional border.

        Args:
            segment: numpy array of shape Nx2, containing all the points in the polygon.
            color: color of the polygon.
            edge_color: color of the polygon edges.
            alpha (float): blending efficient.

        Returns:
            output (VisImage): image object with polygon drawn.
        """
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(
                    color, brightness_factor=-0.7
                )
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    # ----------------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------------

    def _jitter(self, color):
        """Return an RGB color perturbed by random positive noise and clipped to [0, 1]."""
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _create_grayscale_image(self, mask=None):
        """Return a grayscale RGB image, preserving source colors where mask is true."""
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack([img_bw] * 3, axis=2)
        if mask is not None:
            img_bw[mask] = self.img[mask]
        return img_bw

    def _change_color_brightness(self, color, brightness_factor):
        """Adjust HLS lightness by a factor in [-1, 1] and return RGB."""
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

    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes):
            return boxes.tensor.detach().numpy()
        else:
            return np.asarray(boxes)

    def _convert_masks(self, masks_or_polygons):
        """
        Convert different format of masks or polygons to a tuple of masks and polygons.

        Returns:
            list[GenericMask]:
        """

        m = masks_or_polygons
        if isinstance(m, PolygonMasks):
            m = m.polygons
        if isinstance(m, BitMasks):
            m = m.tensor.numpy()
        if isinstance(m, torch.Tensor):
            m = m.numpy()
        ret = []
        for x in m:
            if isinstance(x, GenericMask):
                ret.append(x)
            else:
                ret.append(GenericMask(x, self.output.height, self.output.width))
        return ret

    def _draw_number_in_box(self, box, text, color, label_mode="1"):
        """
        Find proper places to draw text given a box.
        """
        x0, y0, x1, y1 = box
        text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
        horiz_align = "left"
        # for small objects, draw text at the side to avoid occlusion
        instance_area = (y1 - y0) * (x1 - x0)
        if (
            instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
            or y1 - y0 < 40 * self.output.scale
        ):
            if y1 >= self.output.height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)

        height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
        lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
            * 0.65
            * self._default_font_size
        )
        if label_mode == "a":
            text = self.number_to_string(int(text))
        else:
            text = text
        self.draw_text(
            text,
            text_pos,
            color=lighter_color,
            horizontal_alignment=horiz_align,
            font_size=font_size,
        )

        return str(text)

    @staticmethod
    def number_to_string(n):
        """Convert a positive one-based index to lowercase alphabetic labels: a, ..., z, aa."""
        chars = []
        while n:
            n, remainder = divmod(n - 1, 26)
            chars.append(chr(97 + remainder))
        return "".join(reversed(chars))

    def _draw_number_in_mask(
        self, binary_mask, text, color, added_positions=None, label_mode="1"
    ):
        """
        Find proper places to draw text given a binary mask.
        """
        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), "constant")
        mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)  # coords is [y, x]

        if label_mode == "a":
            text = self.number_to_string(int(text))
        else:
            text = text

        text_position = (
            coords_x[len(coords_x) // 2] + 2,
            coords_y[len(coords_y) // 2] - 6,
        )
        self.draw_text(
            text,
            text_position,
            added_positions=added_positions,
            binary_mask=binary_mask,
            color=color,
        )

        return str(text), text_position

    def _draw_text_in_mask(self, binary_mask, text, color):
        """
        Find proper places to draw text given a binary mask.
        """
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, 8
        )
        if stats[1:, -1].size == 0:
            return
        largest_component_id = np.argmax(stats[1:, -1]) + 1

        # draw text on the largest component, as well as other very large components.
        for cid in range(1, _num_cc):
            if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                bottom = np.max((cc_labels == cid).nonzero(), axis=1)[::-1]
                center[1] = bottom[1] + 2
                self.draw_text(text, center, color=color)

    def get_output(self):
        """Return the current VisImage canvas.

        Returns:
            output (VisImage): the image output containing the visualizations added
            to the image.
        """
        return self.output
