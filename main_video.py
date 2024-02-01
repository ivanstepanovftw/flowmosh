import logging
import os

import cv2
import inverse_optical_flow
import numpy as np
import qoi
from scipy.ndimage import map_coordinates
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def alpha_composite(im1, im2, opacity1 = 1.0, opacity2 = 1.0):
    # Validate the opacity values
    if not 0 <= opacity1 <= 1 or not 0 <= opacity2 <= 1:
        raise ValueError('Opacity must be between 0 and 1')

    # Scale the alpha channels by the provided opacity values
    im1[..., 3] = im1[..., 3] * opacity1
    im2[..., 3] = im2[..., 3] * opacity2

    # Normalize the alpha channels to be between 0 and 1
    im1_alpha = im1[..., 3] / 255.0
    im2_alpha = im2[..., 3] / 255.0

    # Compute the composite alpha channel
    composite_alpha = im1_alpha + im2_alpha * (1 - im1_alpha)

    # Handle case where composite_alpha is 0 to avoid divide by zero error
    mask = composite_alpha > 0
    composite_alpha = np.where(mask, composite_alpha, 1)

    # Compute the composite image
    composite_image = np.empty_like(im1)
    for channel in range(3):
        composite_image[..., channel] = (
            im1[..., channel] * im1_alpha
            + im2[..., channel] * im2_alpha * (1 - im1_alpha)
        ) / composite_alpha

    # Add the composite alpha channel to the image
    composite_image[..., 3] = composite_alpha * 255

    return composite_image.astype(np.uint8)


def warp(image: np.ndarray, backward_flow: np.ndarray, order=3) -> np.ndarray:
    channels, height, width = image.shape
    index_grid = np.mgrid[0:height, 0:width].astype(float)
    # Widely, first channel is horizontal x-axis flow, the second channel is vertical y-axis flow.
    coordinates = index_grid + backward_flow[::-1]
    remapped = np.empty(image.shape, dtype=image.dtype)
    for i in range(channels):
        remapped[i] = map_coordinates(image[i], coordinates, order=order, mode='constant', cval=0)
    return remapped


def flowmosh_forward(styled_bgra, image2_bgra, forward_flow, overshoot=1.0, opacity=1.0, interpolation_points=0, order=3) -> np.ndarray:
    interpolation_factor = overshoot / (interpolation_points + 1)
    current_flow = forward_flow * interpolation_factor

    for i in range(interpolation_points + 1):
        forward_flow_inv, disocclusion_mask = inverse_optical_flow.max_method(current_flow)
        # forward_flow_inv, disocclusion_mask = inverse_optical_flow.avg_method(current_flow)

        styled_bgra = warp(styled_bgra.transpose(2, 0, 1), forward_flow_inv, order=order).transpose(1, 2, 0)
        if i < interpolation_points:
            current_flow = warp(current_flow, forward_flow_inv, order=order)

        styled_bgra = alpha_composite(styled_bgra, image2_bgra, opacity)
        styled_bgra[..., 3] = 255

    return styled_bgra


def flowmosh_backward(styled_bgra, image2_bgra, backward_flow, overshoot=1.0, opacity=1.0, interpolation_points=0, order=3) -> np.ndarray:
    for i in range(interpolation_points + 1):
        current_flow = backward_flow * (i + 1) * overshoot / (interpolation_points + 1)

        styled_bgra = warp(styled_bgra.transpose(2, 0, 1), current_flow, order=order).transpose(1, 2, 0)

        styled_bgra = alpha_composite(styled_bgra, image2_bgra, opacity)
        styled_bgra[..., 3] = 255

    return styled_bgra


def main():
    root_path = "root/sintel.mp4"
    frames_path = f"{root_path}/frame"

    frame_indices = sorted([int(x.name) for x in os.scandir(frames_path) if x.is_dir()])

    styled_prev = None

    with logging_redirect_tqdm():
        for frame_i in tqdm(frame_indices, desc="Flowmoshing"):
            frame_path = f"{frames_path}/{frame_i}"
            flow_path = f"{root_path}/flow"

            content_filepath = f"{frame_path}/content.qoi"
            styled_filepath = f"{frame_path}/styled.qoi"
            forward_flow_filepath = f"{flow_path}/flow_{frame_i-1}_to_{frame_i}.npy"
            backward_flow_filepath = f"{flow_path}/flow_{frame_i}_to_{frame_i-1}.npy"

            if styled_prev is None:
                content = qoi.read(content_filepath)
                styled = content.copy()
                styled = cv2.cvtColor(styled, cv2.COLOR_BGR2BGRA)
            else:
                styled = styled_prev.copy()
                # backward_flow = np.load(backward_flow_filepath)
                # styled = warp(styled.transpose(2, 0, 1), backward_flow, order=3).transpose(1, 2, 0)
                # styled = flowmosh_backward(styled, styled_prev, backward_flow, overshoot=0.5, opacity=1.0, interpolation_points=0, order=3)
                forward_flow = np.load(forward_flow_filepath)
                styled = flowmosh_forward(styled, styled_prev, forward_flow, overshoot=1.0, opacity=1.0, interpolation_points=3, order=3)

            _ = qoi.write(styled_filepath, styled.copy())
            styled_prev = styled.copy()


if __name__ == '__main__':
    main()
