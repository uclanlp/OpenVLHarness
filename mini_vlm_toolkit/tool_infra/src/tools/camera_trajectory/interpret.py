"""DA3 camera-motion interpretation helpers.

Pure numpy/math (+ optional matplotlib for the bird's-eye-view render). Converts
per-image world-to-camera extrinsics into a model-facing description rooted in
the first image's camera frame, using the DA3/OpenCV convention:
    +X = right, +Y = down, +Z = forward.
"""

import base64
import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SALIENT_MOVEMENT = 1.0
SALIENT_ROTATION = 10.0


def coordinate_system_text(reference_image_id: str) -> str:
    return (
        f"All vectors are relative to {reference_image_id}'s camera frame, "
        "using the coordinate system: +X = right, +Y = down, "
        "+Z = forward. With this convention, positive yaw (rotation around Y axis) = right and "
        "positive pitch (rotation around X axis) = down. If the original question assumes a different "
        "coordinate system (e.g. with +Y = up and -Z = forward, which flips "
        "the yaw/pitch signs), convert before answering."
    )


def camera_center_from_w2c(extrinsic: np.ndarray) -> np.ndarray:
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return -R.T @ t


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def compute_yaw_pitch(forward: np.ndarray) -> Tuple[float, float]:
    f = normalize(forward)
    x, y, z = float(f[0]), float(f[1]), float(f[2])
    yaw_deg = math.degrees(math.atan2(x, z))
    pitch_deg = math.degrees(math.atan2(y, math.sqrt(x * x + z * z)))
    return yaw_deg, pitch_deg


def _movement_axis_fragment(axis: str, value: float) -> str:
    if axis == "x":
        word = "right" if value > 0 else "left"
    elif axis == "y":
        word = "down" if value > 0 else "up"
    else:
        word = "forward" if value > 0 else "backward"
    return f"{word} ({value:+.2f})"


def summarize_movement(delta: Sequence[float]) -> str:
    x, y, z = float(delta[0]), float(delta[1]), float(delta[2])
    frag_x = _movement_axis_fragment("x", x)
    frag_y = _movement_axis_fragment("y", y)
    frag_z = _movement_axis_fragment("z", z)
    axes = (("x", x, frag_x), ("y", y, frag_y), ("z", z, frag_z))
    _, _, dom_frag = max(axes, key=lambda item: abs(item[1]))
    salient_fragments = [f for _, v, f in axes if abs(v) >= SALIENT_MOVEMENT]
    if salient_fragments:
        salient_text = "salient direction: " + ", ".join(salient_fragments)
    else:
        salient_text = "salient direction: none (all movements < 1 meter)"
    return f"{salient_text}; detailed: {frag_x}, {frag_y}, {frag_z}"


def _yaw_fragment(yaw_deg: float) -> str:
    direction = "right" if yaw_deg > 0 else "left"
    return f"{direction} (yaw {yaw_deg:+.1f}°)"


def _pitch_fragment(pitch_deg: float) -> str:
    direction = "down" if pitch_deg > 0 else "up"
    return f"{direction} (pitch {pitch_deg:+.1f}°)"


def summarize_rotation(yaw_deg: float, pitch_deg: float) -> str:
    yaw_frag = _yaw_fragment(yaw_deg)
    pitch_frag = _pitch_fragment(pitch_deg)
    dom_frag = yaw_frag if abs(yaw_deg) >= abs(pitch_deg) else pitch_frag
    salient_fragments = []
    if abs(yaw_deg) >= SALIENT_ROTATION:
        salient_fragments.append(yaw_frag)
    if abs(pitch_deg) >= SALIENT_ROTATION:
        salient_fragments.append(pitch_frag)
    if salient_fragments:
        salient_text = "salient direction: " + ", ".join(salient_fragments)
    else:
        salient_text = "salient direction: none (all rotations < 10°)"
    
    return f"{salient_text}; detailed: {yaw_frag}, {pitch_frag}"


def round_list(v: Sequence[float], ndigits: int = 3) -> List[float]:
    arr = np.asarray(v, dtype=float)
    return np.round(arr, ndigits).tolist()


def interpret_camera_trajectory(
    extrinsics: np.ndarray,
    image_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(
            f"extrinsics must have shape (N, 3, 4); got {extrinsics.shape}"
        )
    n = extrinsics.shape[0]
    if image_ids is None:
        image_ids = [f"image_{i}" for i in range(n)]
    image_ids = list(image_ids)
    if len(image_ids) != n:
        raise ValueError(
            f"image_ids length {len(image_ids)} does not match extrinsics {n}"
        )

    R0 = extrinsics[0, :3, :3]
    C0 = camera_center_from_w2c(extrinsics[0])

    outputs: List[Dict[str, Any]] = []
    for i, E in enumerate(extrinsics):
        R = E[:3, :3]
        C = camera_center_from_w2c(E)

        delta_world = C - C0
        delta_ref = R0 @ delta_world
        forward_ref = normalize(R0 @ R[2])
        yaw_deg, pitch_deg = compute_yaw_pitch(forward_ref)

        if i == 0:
            movement_summary = "reference camera"
            rotation_summary = "reference view"
        else:
            movement_summary = summarize_movement(delta_ref)
            rotation_summary = summarize_rotation(yaw_deg, pitch_deg)

        outputs.append(
            {
                "image_id": image_ids[i],
                "image_index": i,
                "camera_center_delta": delta_ref.astype(float).tolist(),
                "forward_direction": forward_ref.astype(float).tolist(),
                "movement_summary": movement_summary,
                "rotation_summary": rotation_summary,
                "yaw_deg": float(yaw_deg),
                "pitch_deg": float(pitch_deg),
            }
        )
    return outputs


def render_camera_motion_message(
    outputs: Sequence[Dict[str, Any]],
    *,
    reference_image_id: str,
) -> str:
    lines: List[str] = ["Camera trajectory summary:"]
    for o in outputs:
        i = o["image_index"]
        image_id = o.get("image_id") or f"image_{i}"
        if i == 0:
            lines.append(f"- {image_id}: reference camera and reference view.")
            continue
        lines.append(f"- {image_id}:")
        lines.append(f"    camera movement: {o['movement_summary']}")
        lines.append(f"    camera rotation: {o['rotation_summary']}")
    lines.append("If no salient movement or rotation is reported, use the magnitude of each axis to determine the most possible movement/rotation direction.")
    lines.extend(["", coordinate_system_text(reference_image_id)])
    return "\n".join(lines)


def render_bird_eye_view_image(
    outputs: Sequence[Dict[str, Any]],
    *,
    reference_image_id: str,
    draw_motion_arrows: bool = True,
) -> Optional[str]:
    """Render a top-down (X-Z) map of camera positions and headings.

    Returns base64-PNG (no data: prefix), or None if matplotlib isn't available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if not outputs:
        return None

    xs = [float(o["camera_center_delta"][0]) for o in outputs]
    zs = [float(o["camera_center_delta"][2]) for o in outputs]
    fxs = [float(o["forward_direction"][0]) for o in outputs]
    fzs = [float(o["forward_direction"][2]) for o in outputs]

    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=120)

    span_x = max(xs) - min(xs)
    span_z = max(zs) - min(zs)
    span = max(span_x, span_z, 1.0)
    arrow_len = 0.18 * span

    palette = plt.get_cmap("tab10").colors
    for idx, o in enumerate(outputs):
        label = o.get("image_id") or f"image_{idx}"
        color = palette[idx % len(palette)]
        marker = "*" if idx == 0 else "o"
        size = 180 if idx == 0 else 110
        ax.scatter([xs[idx]], [zs[idx]], c=[color], s=size, marker=marker,
                   edgecolors="black", linewidths=0.6, label=label, zorder=3)
        fx, fz = fxs[idx], fzs[idx]
        norm = math.hypot(fx, fz)
        ux, uz = (fx / norm, fz / norm) if norm > 1e-8 else (0.0, 1.0)
        ax.annotate(
            label, (xs[idx], zs[idx]),
            textcoords="offset points",
            xytext=(-ux * 12, -uz * 12),
            ha=("right" if ux > 0.3 else "left" if ux < -0.3 else "center"),
            va=("top" if uz > 0.3 else "bottom" if uz < -0.3 else "center"),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.8),
            zorder=5,
        )
        dot_radius = 0.04 * span if idx == 0 else 0.03 * span
        start_x = xs[idx] + ux * dot_radius
        start_z = zs[idx] + uz * dot_radius
        usable_len = max(arrow_len - dot_radius, 0.001)
        ax.quiver(
            start_x, start_z,
            ux * usable_len, uz * usable_len,
            angles="xy", scale_units="xy", scale=1,
            color=color,
            width=0.008,
            headwidth=4.5,
            headlength=6,
            headaxislength=5,
            pivot="tail",
            zorder=4,
        )

    if draw_motion_arrows and len(outputs) > 1:
        for idx in range(1, len(outputs)):
            ax.annotate(
                "", xy=(xs[idx], zs[idx]), xytext=(xs[idx-1], zs[idx-1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="dimgray",
                    alpha=0.85,
                    linewidth=1.5,
                    linestyle="dashed",
                    shrinkA=10, shrinkB=10,
                ),
                zorder=2,
            )

    pad = 0.25 * span
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(zs) - pad, max(zs) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("+X (right)")
    ax.set_ylabel("+Z (forward)")
    ax.set_title(f"Bird's-eye view (in {reference_image_id}'s camera frame)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
