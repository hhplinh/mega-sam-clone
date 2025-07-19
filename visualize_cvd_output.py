import argparse
import asyncio
from pathlib import Path

import numpy as np
import viser


def load_cvd_output(path: Path):
    data = np.load(path)
    images = data["images"]
    depths = data["depths"]
    intrinsic = data["intrinsic"]
    cam_c2w = data["cam_c2w"]
    return images, depths, intrinsic, cam_c2w


def depth_to_pointcloud(depth: np.ndarray, rgb: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pixels = np.stack([xs, ys, np.ones_like(xs)], axis=-1).reshape(-1, 3)
    invK = np.linalg.inv(K)
    cam = (invK @ pixels.T) * depth.reshape(1, -1)
    cam_h = np.vstack([cam, np.ones((1, cam.shape[1]))])
    world = (c2w @ cam_h)[:3].T
    colors = rgb.reshape(-1, 3) / 255.0
    return world.astype(np.float32), colors.astype(np.float32)


async def run_server(args):
    images, depths, K, cam_c2w = load_cvd_output(args.input)
    n_frames = images.shape[0]

    pcds = []
    cols = []
    for i in range(n_frames):
        p, c = depth_to_pointcloud(depths[i], images[i], K, cam_c2w[i])
        pcds.append(p)
        cols.append(c)

    server = viser.ViserServer()

    slider = server.add_gui_slider("frame", min=0, max=n_frames - 1, step=1)
    play_button = server.add_gui_button("Play/Pause")

    point_cloud = server.add_point_cloud(points=pcds[0], colors=cols[0])
    camera = server.add_camera_frustum(transform=cam_c2w[0])

    playing = False

    @slider.on_update
    def on_slider(event):
        idx = int(event.value)
        point_cloud.update(points=pcds[idx], colors=cols[idx])
        camera.update(transform=cam_c2w[idx])

    @play_button.on_click
    async def on_play(event):
        nonlocal playing
        playing = not playing

    idx = 0
    while True:
        await asyncio.sleep(0.1)
        if playing:
            idx = (idx + 1) % n_frames
            slider.value = idx
            point_cloud.update(points=pcds[idx], colors=cols[idx])
            camera.update(transform=cam_c2w[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cvd_opt output")
    parser.add_argument("input", type=Path, help="npz file produced by cvd_opt.py")
    args = parser.parse_args()

    asyncio.run(run_server(args))
