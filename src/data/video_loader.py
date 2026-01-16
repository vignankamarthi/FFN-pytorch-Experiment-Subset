"""Video loading utilities using PyAV.

This module handles loading video frames from .webm files and implements
uniform frame sampling for temporal models.
"""

from typing import List, Optional

import av
import numpy as np


def sample_frame_indices(total_frames: int, num_segments: int) -> List[int]:
    """
    Sample frame indices using uniform segment sampling.

    Divides the video into num_segments equal segments and samples
    the center frame of each segment. This is the standard approach
    for temporal action recognition.

    Parameters
    ----------
    total_frames : int
        Total number of frames in the video.
    num_segments : int
        Number of frames to sample (T in the model).

    Returns
    -------
    List[int]
        List of frame indices to extract.

    Examples
    --------
    >>> sample_frame_indices(48, 8)
    [3, 9, 15, 21, 27, 33, 39, 45]
    """
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")

    # Handle case where video has fewer frames than requested segments
    if total_frames < num_segments:
        # Repeat last frame to fill
        indices = list(range(total_frames))
        indices.extend([total_frames - 1] * (num_segments - total_frames))
        return indices

    # Segment length
    seg_len = total_frames / num_segments

    # Sample center of each segment
    indices = []
    for i in range(num_segments):
        start = seg_len * i
        end = seg_len * (i + 1)
        center = int((start + end) / 2)
        # Clamp to valid range
        center = min(center, total_frames - 1)
        indices.append(center)

    return indices


def load_video_frames(
    video_path: str,
    num_frames: int,
    frame_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Load specific frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file (.webm).
    num_frames : int
        Number of frames to load (T).
    frame_indices : Optional[List[int]]
        Specific frame indices to load. If None, uses uniform sampling.

    Returns
    -------
    np.ndarray
        Array of shape (T, H, W, 3) containing RGB frames.

    Raises
    ------
    FileNotFoundError
        If video file does not exist.
    ValueError
        If video has no frames.
    """
    # Open video container
    try:
        container = av.open(video_path)
    except av.error.FileNotFoundError:
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Decode all frames (necessary for webm since seeking is unreliable)
    all_frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        all_frames.append(img)

    container.close()

    total_frames = len(all_frames)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    # Get frame indices if not provided
    if frame_indices is None:
        frame_indices = sample_frame_indices(total_frames, num_frames)

    # Extract requested frames
    frames = []
    for idx in frame_indices:
        # Clamp index to valid range
        idx = max(0, min(idx, total_frames - 1))
        frames.append(all_frames[idx])

    return np.stack(frames, axis=0)


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    dict
        Dictionary with 'num_frames', 'height', 'width' keys.
    """
    container = av.open(video_path)

    num_frames = 0
    height = 0
    width = 0

    for frame in container.decode(video=0):
        if num_frames == 0:
            height = frame.height
            width = frame.width
        num_frames += 1

    container.close()

    return {
        "num_frames": num_frames,
        "height": height,
        "width": width,
    }
