# Camera trajectory wrapper

This implementation runs inside the common router and calls the same DA3 raw
process as depth estimation. Follow [DA3 setup](../depth_estimation/README.md)
and configure `services.tools.camera_trajectory.server_url` to that process.
The paper YAML shares the depth endpoint configuration.

The public tool takes ordered `image_ids`. The capability resolves them to
bytes and preserves their order. The first image is the reference camera;
numeric evidence uses +X right, +Y down, +Z forward in that reference frame.
Raw intrinsics and world-to-camera extrinsics become float32 arrays in memory.

The bird's-eye view is drawn in the **router environment**. The server extra installs Matplotlib
there to reproduce the paper evidence. A manually assembled router environment
also needs `python -m pip install matplotlib` for this renderer. When Matplotlib is unavailable,
the wrapper keeps numeric/text results and omits the image. No second camera
GPU process is needed. The depth/camera gate is shared within one router.
