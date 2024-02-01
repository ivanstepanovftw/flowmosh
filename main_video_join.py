import os

import cv2
import numpy as np
import qoi
from torchvision import transforms
from tqdm import tqdm


def content_to_frame(ndarray: np.ndarray) -> np.ndarray:
    """
    Input: (H, W, C) ndarray, RGB or RGBA
    Output: (H, W, 3) ndarray, BGR
    """
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x[..., [2, 1, 0]]),  # RGB to BGR
    ])
    return transform(ndarray)


def main():
    video_filepath = "examples/video/sintel.mp4"
    root_path = "root/sintel.mp4"
    frames_path = f"{root_path}/frame"
    output_filepath, fourcc = "output/sintel.mp4/out_mp4v.mp4", "mp4v"
    # output_filepath, fourcc = "output/sintel.mp4/out_vp90.webm", "VP90"
    # output_filepath, fourcc = "output/sintel.mp4/out_ffv1.avi", "FFV1"
    # output_filepath, fourcc = "output/sintel.mp4/out_xvid.avi", "xvid"
    # output_filepath, fourcc = "output/sintel.mp4/out_mjpg.avi", "MJPG"

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    vidcap = cv2.VideoCapture(video_filepath)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted([int(x.name) for x in os.scandir(frames_path) if x.is_dir()])
    if len(frame_indices) != frame_count:
        print("WARNING: Number of frames in video does not match number of frames in root directory")
    fourcc = cv2.VideoWriter_fourcc(*fourcc)

    out = None

    for frame_i in tqdm(frame_indices, desc="Joining video"):
        frame_path = f"{frames_path}/{frame_i}"

        styled_filepath = f"{frame_path}/styled.qoi"
        styled = qoi.read(styled_filepath)
        styled = content_to_frame(styled)
        h, w, c = styled.shape
        if c == 4:
            styled = styled[..., :3]
        if out is None:
            frame_shape = w, h
            out = cv2.VideoWriter(output_filepath, fourcc, vidcap.get(cv2.CAP_PROP_FPS), frame_shape)
        out.write(styled)

    if out is not None:
        out.release()


if __name__ == '__main__':
    main()
