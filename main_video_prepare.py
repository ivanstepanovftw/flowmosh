import os

import cv2
import numpy as np
import qoi
import torch
import torchvision
import torchvision.models.optical_flow
from torchvision import transforms
from tqdm import tqdm


class DivisibleBy8:
    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """ Preprocesses a tensor to be divisible by 8. This is required by the RAFT model. """
        h, w = x.shape[:2]
        h = h - h % 8
        w = w - w % 8
        x = x[:h, :w, ...]
        return x


def frame_to_content(frame: np.ndarray) -> np.ndarray:
    """
    Input: (H, W, C) ndarray, BGR or BGRA
    Output: (H, W, 3) ndarray, RGB
    """
    transform = transforms.Compose([
        DivisibleBy8(),
        torchvision.transforms.Lambda(lambda x: x[..., [2, 1, 0]]),  # BGR to RGB
    ])
    return transform(frame)


def frame_to_tensor(frame: np.ndarray, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Input: (H, W, C) ndarray, BGR or BGRA
    Output: (C, H, W) Tensor, normalized
    """
    transform = transforms.Compose([
        DivisibleBy8(),
        torchvision.transforms.Lambda(lambda x: x[..., [2, 1, 0]]),  # BGR to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(frame)


def main():
    video_filepath = 'examples/video/sintel.mp4'
    root_path = "root/sintel.mp4"

    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.optical_flow.raft_large(weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2).to(device)
    # model = torchvision.models.optical_flow.raft_small(weights=torchvision.models.optical_flow.Raft_Small_Weights.C_T_V2).to(device)
    model.eval()

    raft_mean = torch.tensor((0.5, 0.5, 0.5)).to(device)
    raft_std = torch.tensor((0.5, 0.5, 0.5)).to(device)

    vidcap = cv2.VideoCapture(video_filepath)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    content_prev = None

    for frame_idx in tqdm(range(frame_count), desc="Splitting video and computing optical flow"):
        frame_path = f"{root_path}/frame/{frame_idx}"
        flow_path = f"{root_path}/flow"

        os.makedirs(frame_path, exist_ok=True)
        os.makedirs(flow_path, exist_ok=True)

        content_filepath = f"{frame_path}/content.qoi"
        forward_flow_filepath = f"{flow_path}/flow_{frame_idx-1}_to_{frame_idx}.npy"
        backward_flow_filepath = f"{flow_path}/flow_{frame_idx}_to_{frame_idx-1}.npy"

        success, frame_cur = vidcap.read()
        if not success:
            raise Exception(f"Failed to read frame {frame_idx} from video of {frame_count} frames")

        content = frame_to_content(frame_cur).copy()
        _ = qoi.write(content_filepath, content)

        content_cur = frame_to_tensor(frame_cur, raft_mean, raft_std).to(device)

        if content_prev is not None:
            if not os.path.exists(forward_flow_filepath):
                flow: list[torch.Tensor] = model(content_prev[None], content_cur[None])
                np.save(forward_flow_filepath, flow[0][0].cpu().detach().numpy())

            if not os.path.exists(backward_flow_filepath):
                flow: list[torch.Tensor] = model(content_cur[None], content_prev[None])
                np.save(backward_flow_filepath, flow[0][0].cpu().detach().numpy())

        content_prev = content_cur


if __name__ == '__main__':
    main()
