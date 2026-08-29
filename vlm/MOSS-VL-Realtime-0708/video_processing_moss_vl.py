# coding=utf-8
# Copyright 2025 The FNLP Vision Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""video processor class for Moss-VL."""

import json
import logging as system_logging
import math
import os
import re
import subprocess
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from joblib import Parallel, delayed
from torchcodec.decoders import VideoDecoder

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size, validate_kwargs
from transformers.processing_utils import Unpack, VideosKwargs
from transformers.utils import TensorType, add_start_docstrings, logging
from transformers.video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from transformers.video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


logger = logging.get_logger(__name__)

TORCHCODEC_TIMESTAMP_EPSILON = 1e-6


# -----------------------------------------------------------------------------
# Torchcodec video frame extraction utilities
# -----------------------------------------------------------------------------

def check_video_for_extra_streams_and_errors(video_path: str) -> dict:
    """
    Check if video file has abnormal streams or errors reported by ffprobe.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        A dictionary containing:
        - 'has_extra_streams': bool, whether there are streams other than video and audio.
        - 'unsupported_codec_errors': list, all "Unsupported codec" error messages.
        - 'ffprobe_output_error': str, other errors/warnings from ffprobe stderr.
        - 'ffprobe_successful': bool, whether ffprobe command executed successfully (return code 0).
        - 'stream_details': list, codec_type and index for each stream.
        - 'num_streams': int, total number of streams identified in the video file.
    """
    result = {
        'has_extra_streams': False,
        'unsupported_codec_errors': [],
        'ffprobe_output_error': '',
        'ffprobe_successful': False,
        'stream_details': [],
        'num_streams': 0
    }
    
    command = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        video_path
    ]
    
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        result['ffprobe_successful'] = (process.returncode == 0)
        
        if process.stderr:
            result['ffprobe_output_error'] = process.stderr
            unsupported_codec_pattern = re.compile(r"Unsupported codec with id \d+ for input stream \d+")
            result['unsupported_codec_errors'] = unsupported_codec_pattern.findall(process.stderr)
        
        if process.stdout:
            ffprobe_data = json.loads(process.stdout)
            if 'streams' in ffprobe_data:
                result['num_streams'] = len(ffprobe_data['streams'])
                for stream in ffprobe_data['streams']:
                    stream_type = stream.get('codec_type')
                    stream_index = stream.get('index')
                    result['stream_details'].append({'index': stream_index, 'codec_type': stream_type})
                    if stream_type not in ['video', 'audio']:
                        result['has_extra_streams'] = True
                        
            if 'format' in ffprobe_data and 'nb_streams' in ffprobe_data['format']:
                if result['num_streams'] == 0:
                    result['num_streams'] = ffprobe_data['format']['nb_streams']
                elif result['num_streams'] != ffprobe_data['format']['nb_streams']:
                    logger.warning(
                        f"Number of streams in 'streams' list ({result['num_streams']}) "
                        f"differs from 'nb_streams' in 'format' ({ffprobe_data['format']['nb_streams']})."
                    )
    except FileNotFoundError:
        result['ffprobe_output_error'] = "ffprobe command not found. Please ensure FFmpeg is installed and in your PATH."
        result['ffprobe_successful'] = False
    except json.JSONDecodeError:
        result['ffprobe_output_error'] = "Failed to parse ffprobe JSON output. Check ffprobe installation or video file."
        result['ffprobe_successful'] = False
    except Exception as e:
        result['ffprobe_output_error'] = f"An unexpected error occurred: {e}"
        result['ffprobe_successful'] = False
        
    return result


def remove_video_extra_stream_ffmpeg(input_video: str, output_video: str) -> bool:
    """
    Remove extra streams from video using ffmpeg.
    
    Args:
        input_video: Path to input video.
        output_video: Path to output video.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    command_list = [
        "ffmpeg", "-y", "-i", input_video,
        "-map", "0:v:0",
        "-c", "copy",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata", "-1",
        "-map_chapters", "-1",
        "-movflags", "faststart",
        output_video,
    ]
    
    try:
        subprocess.run(command_list, shell=False, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        system_logging.error(f"Command execution failed with return code: {e.returncode}, video: {input_video}")
        system_logging.error(f"Error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        system_logging.error("Error: ffmpeg command not found. Please ensure ffmpeg is installed and in PATH.")
        return False
    except Exception as e:
        system_logging.error(f"Unexpected error executing command: {e}, video: {input_video}", exc_info=True)
        return False


def clean_video_streams(video_path: str) -> str:
    """
    Clean video streams if extra streams are detected.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        str: Path to cleaned video (or original if no cleaning needed).
    """
    ffprobe_res = check_video_for_extra_streams_and_errors(video_path)
    if ffprobe_res['has_extra_streams']:
        base_name = os.path.basename(video_path)
        output_folder = os.path.dirname(video_path)
        file_name_without_ext, file_ext = os.path.splitext(base_name)
        new_base_name = f"{file_name_without_ext}_fix{file_ext}"
        video_path_output = os.path.join(output_folder, new_base_name)
        
        process_flag = remove_video_extra_stream_ffmpeg(video_path, video_path_output)
        if not process_flag:
            logger.warning("Failed to remove extra streams with ffmpeg")
            return video_path
        return video_path_output
    return video_path


@lru_cache(maxsize=8192)
def cached_clean_video_streams(video_path: str) -> str:
    return clean_video_streams(video_path)


def clamp_timestamps_for_torchcodec(timestamps: List[float], torchcodec_metadata) -> List[float]:
    if not timestamps:
        return timestamps

    min_pts = torchcodec_metadata.begin_stream_seconds_from_content
    if min_pts is None:
        min_pts = 0.0

    max_pts_candidates = []
    if torchcodec_metadata.num_frames_from_content and torchcodec_metadata.average_fps:
        max_pts_candidates.append(
            (torchcodec_metadata.num_frames_from_content - 1) / torchcodec_metadata.average_fps + min_pts
        )
    if torchcodec_metadata.end_stream_seconds_from_content is not None:
        # TorchCodec requires requested PTS to be strictly smaller than the content end.
        max_pts_candidates.append(torchcodec_metadata.end_stream_seconds_from_content - TORCHCODEC_TIMESTAMP_EPSILON)
    if not max_pts_candidates and torchcodec_metadata.duration_seconds is not None:
        max_pts_candidates.append(torchcodec_metadata.duration_seconds - TORCHCODEC_TIMESTAMP_EPSILON)

    if max_pts_candidates:
        max_pts = max(min_pts, min(max_pts_candidates))
        return [max(min_pts, min(float(t), max_pts)) for t in timestamps]
    if min_pts > 0:
        return [max(min_pts, float(t)) for t in timestamps]
    return [float(t) for t in timestamps]


def split_indices(indices: List[Union[int, float]], num_chunks: int) -> List[List[Union[int, float]]]:
    """
    Split an index list into roughly equal chunks.
    
    Args:
        indices: List of indices to split.
        num_chunks: Number of chunks to create.
        
    Returns:
        List of index chunks.
    """
    chunk_size = len(indices) // num_chunks
    chunks = []
    for i in range(num_chunks - 1):
        chunks.append(indices[i * chunk_size:(i + 1) * chunk_size])
    chunks.append(indices[(num_chunks - 1) * chunk_size:])
    return chunks


def decode_sequentially(indices: List[int], video_path: str, ffmpeg_threads: int = 0):
    """
    Decode frames sequentially from a video.
    
    Args:
        indices: List of frame indices to decode.
        video_path: Path to the video file.
        ffmpeg_threads: Number of ffmpeg threads to use.
        
    Returns:
        FrameBatch from torchcodec.
    """
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=ffmpeg_threads)
    try:
        return decoder.get_frames_at(indices)
    finally:
        del decoder


def decode_with_multithreading(indices: List[int], num_threads: int, video_path: str) -> dict:
    """
    Decode frames using multithreading with joblib.
    
    Args:
        indices: List of frame indices to decode.
        num_threads: Number of threads to use.
        video_path: Path to the video file.
        
    Returns:
        dict: Contains 'data', 'duration_seconds', 'pts_seconds' tensors.
    """
    chunks = split_indices(indices, num_chunks=num_threads)
    results = Parallel(n_jobs=num_threads, prefer="threads", verbose=0)(
        delayed(decode_sequentially)(chunk, video_path) for chunk in chunks
    )
    
    return {
        "data": torch.cat([frame_batch.data for frame_batch in results], dim=0),
        "duration_seconds": torch.cat([frame_batch.duration_seconds for frame_batch in results], dim=0),
        "pts_seconds": torch.cat([frame_batch.pts_seconds for frame_batch in results], dim=0)
    }


def decode_sequentially_timestamp(timestamp_list: List[float], video_path: str, ffmpeg_threads: int = 0):
    """
    Decode frames sequentially from a video based on timestamps.
    
    Args:
        timestamp_list: List of timestamps (in seconds) to decode.
        video_path: Path to the video file.
        ffmpeg_threads: Number of ffmpeg threads to use.
        
    Returns:
        FrameBatch from torchcodec.
    """
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=ffmpeg_threads)
    try:
        metadata = decoder.metadata
        timestamp_list = clamp_timestamps_for_torchcodec(timestamp_list, metadata)

        return decoder.get_frames_played_at(timestamp_list)
    finally:
        del decoder


def timestamp_decode_with_multithreading(timestamp_list: List[float], num_threads: int, video_path: str) -> dict:
    """
    Decode frames using multithreading based on timestamps.
    
    Args:
        timestamp_list: List of timestamps (in seconds) to decode.
        num_threads: Number of threads to use.
        video_path: Path to the video file.
        
    Returns:
        dict: Contains 'data', 'duration_seconds', 'pts_seconds' tensors.
    """
    chunks = split_indices(timestamp_list, num_chunks=num_threads)
    results = Parallel(n_jobs=num_threads, prefer="threads", verbose=0)(
        delayed(decode_sequentially_timestamp)(chunk, video_path) for chunk in chunks
    )
    
    # Concatenate results from all threads
    data_list = [frame_batch.data for frame_batch in results]
    duration_list = [frame_batch.duration_seconds for frame_batch in results]
    pts_list = [frame_batch.pts_seconds for frame_batch in results]
    
    if not data_list:
        logger.warning("No frames were successfully decoded.")
        return {"data": torch.empty(0), "duration_seconds": torch.empty(0), "pts_seconds": torch.empty(0)}
    
    return {
        "data": torch.cat(data_list, dim=0),
        "duration_seconds": torch.cat(duration_list, dim=0),
        "pts_seconds": torch.cat(pts_list, dim=0)
    }


def extract_frames_with_torchcodec(
    video_path: str,
    sample_frames_count: int,
    num_threads: int = 4,

) -> Optional[dict]:
    """
    Extract frames from video using torchcodec with multithreading.
    
    Args:
        video_path: Path to the video file.
        sample_frames_count: Number of frames to sample.
        num_threads: Number of threads to use for extraction.
        sampling_method: Sampling method, either "index" (uniform frame indices) or "timestamp" (uniform timestamps).
        
    Returns:
        dict: Contains 'data' (N, C, H, W), 'duration_seconds' (N,), 'pts_seconds' (N,) tensors.
              Returns None if extraction fails.
    """
    try:
        video_path = cached_clean_video_streams(video_path)
        decoder = VideoDecoder(video_path, num_ffmpeg_threads=0)
        metadata = decoder.metadata


        total_frames_in_video = metadata.num_frames_from_content
        
        effective_sample_count = min(sample_frames_count, total_frames_in_video)
        if effective_sample_count == 0:
            logger.error("Cannot extract frames: video has 0 frames or specified frame count is 0")
            return None
        
        # Generate uniform frame indices
        frame_indices = np.linspace(0, total_frames_in_video - 1, effective_sample_count).astype(np.int32)
        # Ensure indices are valid and remove duplicates
        frame_indices = np.unique(np.clip(frame_indices, 0, total_frames_in_video - 1))
        
        result = decode_with_multithreading(frame_indices.tolist(), num_threads=num_threads, video_path=video_path)
        # Add frame_indices to the result for later use
        result["frame_indices"] = frame_indices
        return result


            
    except Exception:
        traceback.print_exc()
        return None


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
    per_frame_min_pixels: int = None,
    per_frame_max_pixels: int = None,
):
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor

    # Step 1: Apply per-frame upper limit constraint
    if per_frame_max_pixels is not None and h_bar * w_bar > per_frame_max_pixels:
        beta = math.sqrt((height * width) / per_frame_max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)

    # Step 2: Apply 3D volume constraints (frames * height * width)
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    # Step 3: Ensure per-frame lower limit is respected (after volume constraint)
    # This guarantees single frame stays within [per_frame_min_pixels, per_frame_max_pixels]
    if per_frame_min_pixels is not None and h_bar * w_bar < per_frame_min_pixels:
        beta = math.sqrt(per_frame_min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


class MossVLVideoProcessorInitKwargs(VideosKwargs):
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    min_frames: Optional[int]
    max_frames: Optional[int]
    video_fps: Optional[Union[int, float]]
    num_extract_threads: Optional[int]
    # Total 3D volume budget across all videos; distributed proportionally per video by T*H*W
    video_max_pixels: Optional[int]


@add_start_docstrings(
    "Constructs a fast Moss-VL video processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
        video_fps (`float`, *optional*, defaults to 1.0):
            Target frames per second for video sampling.
        min_frames (`int`, *optional*, defaults to 1):
            Minimum number of frames to sample from a video.
        max_frames (`int`, *optional*, defaults to 256):
            Maximum number of frames to sample from a video.
        num_extract_threads (`int`, *optional*, defaults to 4):
            Number of threads to use for frame extraction.
    """,
)
class MossVLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 32 * 32, "longest_edge": 32 * 32 * 768}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    video_fps = 1.0
    min_frames = 1
    max_frames = 256
    num_extract_threads = 4
    do_sample_frames = True
    # Total 3D volume budget across all videos; distributed proportionally per video by T*H*W
    video_max_pixels = None  # read from config
    valid_kwargs = MossVLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[MossVLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)
        if self.size is not None and (
            self.size.get("shortest_edge", None) is None or self.size.get("longest_edge", None) is None
        ):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

    def _further_process_kwargs(
        self,
        size: Optional[SizeDict] = None,
        **kwargs,
    ) -> dict:
        """
        Update kwargs that need further processing before being validated
        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if size is not None and ("shortest_edge" not in size or "longest_edge" not in size):
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        return super()._further_process_kwargs(size=size, **kwargs)

    def _get_video_path_from_input(self, video_input: Union[str, Dict[str, Any]]) -> str:
        """Normalize a video input into a video path."""
        if isinstance(video_input, dict):
            return video_input["video_path"]
        return video_input

    def _get_video_duration_seconds(self, video_input: Union[str, Dict[str, Any]]) -> float:
        """Get video duration in seconds for weighted frame-budget allocation."""
        video_path = cached_clean_video_streams(self._get_video_path_from_input(video_input))
        decoder = VideoDecoder(video_path, num_ffmpeg_threads=0)
        try:
            metadata = decoder.metadata
            duration = None
            if (
                metadata.end_stream_seconds_from_content is not None
                and metadata.begin_stream_seconds_from_content is not None
            ):
                duration = metadata.end_stream_seconds_from_content - metadata.begin_stream_seconds_from_content
            if duration is None or duration <= 0:
                duration = metadata.duration_seconds
            return max(0.0, float(duration or 0.0))
        finally:
            del decoder

    def _allocate_max_frames_for_multiple_videos(
        self,
        video_inputs: List[Union[str, Dict[str, Any]]],
        total_max_frames: Optional[int],
    ) -> List[Optional[int]]:
        """
        Treat max_frames as a total budget for multi-video input and allocate it by duration.

        The returned values are per-video max_frames. Segment dict inputs still keep their
        existing per-segment weighting logic after receiving the video-level allocation.
        """
        if not video_inputs:
            return []
        if total_max_frames is None or len(video_inputs) == 1:
            return [total_max_frames] * len(video_inputs)

        total_max_frames = int(total_max_frames)
        num_videos = len(video_inputs)
        if total_max_frames < num_videos:
            logger.warning(
                "Received max_frames=%s for %s videos. At least one frame per video is required, "
                "so falling back to 1 frame per video.",
                total_max_frames,
                num_videos,
            )
            return [1] * num_videos

        video_durations = [self._get_video_duration_seconds(video_input) for video_input in video_inputs]
        total_duration = sum(video_durations)

        # Reserve one frame per video first, then distribute the remaining budget by duration.
        allocations = [1] * num_videos
        remaining_budget = total_max_frames - num_videos
        if remaining_budget == 0:
            return allocations

        if total_duration <= 0:
            raw_extra_allocations = [remaining_budget / num_videos] * num_videos
        else:
            raw_extra_allocations = [
                remaining_budget * (duration / total_duration) for duration in video_durations
            ]

        base_extra_allocations = [int(math.floor(value)) for value in raw_extra_allocations]
        allocations = [base + extra for base, extra in zip(allocations, base_extra_allocations)]

        remainder = remaining_budget - sum(base_extra_allocations)
        if remainder > 0:
            fractional_parts = [
                (raw_value - base_value, index)
                for index, (raw_value, base_value) in enumerate(zip(raw_extra_allocations, base_extra_allocations))
            ]
            fractional_parts.sort(key=lambda item: (-item[0], item[1]))
            for _, index in fractional_parts[:remainder]:
                allocations[index] += 1

        return allocations

    def calculate_num_frames(
        self,
        metadata: VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> int:
        """
        Calculate the number of frames to sample using fps-based logic with min/max constraints.
        
        Logic:
        1. Calculate target_frames based on fps and video duration
        2. Apply min_frames and max_frames constraints
        3. Apply max_allowed_frames protection (rough cap from total video_max_pixels budget)
        4. Return the number of frames to sample
        
        Args:
            metadata (`VideoMetadata`):
                Metadata of the video containing information about total duration, fps and total number of frames.
            num_frames (`int`, *optional*):
                Maximum number of frames to sample. If provided, overrides fps-based calculation.
            fps (`int` or `float`, *optional*):
                Target frames to sample per second. Defaults to `self.video_fps`.
            min_frames (`int`, *optional*):
                Minimum number of frames to sample. If None, uses self.min_frames.
            max_frames (`int`, *optional*):
                Maximum number of frames to sample. If None, uses self.max_frames.
        Returns:
            int:
                Number of frames to sample.
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive arguments, please use only one!")

        total_num_frames = metadata.total_num_frames
        
        # Use provided min/max or fall back to defaults
        effective_min_frames = min_frames if min_frames is not None else self.min_frames
        effective_max_frames = max_frames if max_frames is not None else self.max_frames
        
        # Rough per-video frame cap derived from the multi-video total budget
        # (exact allocation happens later in _preprocess via weighted distribution)
        per_frame_min_pixels = self.size.get("shortest_edge", None) if self.size else None
        video_max_pixels = getattr(self, "video_max_pixels", None)
        if per_frame_min_pixels is not None and video_max_pixels is not None and per_frame_min_pixels > 0:
            max_allowed_frames = video_max_pixels // per_frame_min_pixels
            effective_max_frames = min(effective_max_frames, max_allowed_frames)
        
        # Get video duration
        if hasattr(metadata, 'duration') and metadata.duration is not None:
            duration = metadata.duration
        else:
            video_fps = metadata.fps
            if video_fps is not None and video_fps > 0:
                duration = total_num_frames / video_fps
            else:
                # Fallback: assume 24 fps
                video_fps = 24.0
                duration = total_num_frames / video_fps
                logger.warning_once(
                    "Could not determine video fps from metadata, defaulting to 24 fps for duration calculation."
                )

        # Use provided fps or default
        target_fps = fps if fps is not None else self.video_fps
        
        # Calculate target frames based on fps and duration
        if num_frames is None:
            # Calculate how many frames we should sample based on target fps
            target_total_frames = int(math.ceil(duration * target_fps - 1e-6))
            
            # Apply min/max constraints
            sample_frames = max(target_total_frames, effective_min_frames)
            sample_frames = min(sample_frames, effective_max_frames, total_num_frames)
        else:
            # If num_frames is explicitly provided, use it directly with constraints
            sample_frames = min(max(num_frames, effective_min_frames), effective_max_frames, total_num_frames)

        return sample_frames


    def _decode_timestamps_with_decoder(
        self,
        decoder: VideoDecoder,
        timestamps: List[float],
        chunk_size: int = 128,
    ) -> torch.Tensor:
        if not timestamps:
            return torch.empty(0)

        frame_chunks = []
        for start in range(0, len(timestamps), chunk_size):
            frame_batch = decoder.get_frames_played_at(timestamps[start:start + chunk_size])
            frame_chunks.append(frame_batch.data)

        if len(frame_chunks) == 1:
            return frame_chunks[0]
        return torch.cat(frame_chunks, dim=0)

    def _clamp_timestamps_for_decoder(
        self,
        timestamps: List[float],
        torchcodec_metadata,
    ) -> List[float]:
        return clamp_timestamps_for_torchcodec(timestamps, torchcodec_metadata)

    def _fetch_video_segments_batched(
        self,
        video_path: str,
        segments: List[List[float]],
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        video_fps: Optional[float] = None,
    ):
        min_frames = max(1, min_frames if min_frames is not None else self.min_frames)
        max_frames = max(1, max_frames if max_frames is not None else self.max_frames)
        target_video_fps = video_fps if video_fps is not None else self.video_fps

        video_path = cached_clean_video_streams(video_path)
        decoder = VideoDecoder(video_path, num_ffmpeg_threads=0)
        try:
            torchcodec_metadata = decoder.metadata
            source_video_fps = torchcodec_metadata.average_fps

            duration = None
            if (
                torchcodec_metadata.end_stream_seconds_from_content is not None
                and torchcodec_metadata.begin_stream_seconds_from_content is not None
            ):
                duration = (
                    torchcodec_metadata.end_stream_seconds_from_content
                    - torchcodec_metadata.begin_stream_seconds_from_content
                )
            if duration is None or duration <= 0:
                duration = torchcodec_metadata.duration_seconds

            segment_durations = [
                segment[1] - segment[0] if len(segment) == 2 else None
                for segment in segments
            ]
            total_segment_duration = sum(d for d in segment_durations if d is not None)
            num_range_segments = sum(1 for d in segment_durations if d is not None)

            segment_timestamps = []
            decode_timestamps = []
            for i, segment in enumerate(segments):
                if len(segment) == 1:
                    actual_timestamps = self._clamp_timestamps_for_decoder([segment[0]], torchcodec_metadata)
                    segment_timestamps.append(actual_timestamps)
                    decode_timestamps.extend(actual_timestamps)
                    continue

                start_time, end_time = segment
                segment_duration = end_time - start_time
                target_frames = int(math.ceil(segment_duration * target_video_fps))

                if total_segment_duration > 0:
                    weight = segment_durations[i] / total_segment_duration
                else:
                    weight = 1.0 / num_range_segments if num_range_segments > 0 else 1.0

                weighted_min_frames = max(1, int(round(min_frames * weight)))
                weighted_max_frames = max(1, int(round(max_frames * weight)))
                target_frames = max(target_frames, weighted_min_frames)
                target_frames = min(target_frames, weighted_max_frames)

                if target_frames == 1:
                    actual_timestamps = [start_time]
                else:
                    actual_timestamps = np.linspace(
                        start_time,
                        end_time,
                        target_frames,
                        endpoint=False,
                    ).tolist()

                actual_timestamps = self._clamp_timestamps_for_decoder(actual_timestamps, torchcodec_metadata)
                segment_timestamps.append(actual_timestamps)
                decode_timestamps.extend(actual_timestamps)

            flat_frames = self._decode_timestamps_with_decoder(decoder, decode_timestamps)

            videos = []
            metadata = []
            frame_offset = 0
            for actual_timestamps in segment_timestamps:
                sample_count = len(actual_timestamps)
                video_tensor = flat_frames[frame_offset:frame_offset + sample_count]
                frame_offset += sample_count

                video_metadata = VideoMetadata(
                    total_num_frames=sample_count,
                    fps=source_video_fps,
                    duration=duration,
                    video_backend="torchcodec",
                    height=torchcodec_metadata.height,
                    width=torchcodec_metadata.width,
                    frames_indices=None,
                )
                video_metadata.actual_timestamps = actual_timestamps

                videos.append(video_tensor)
                metadata.append(video_metadata)

            return videos, metadata
        finally:
            del decoder


    def _fetch_video_segment(
        self,
        video_path: str,
        segment: List[float],
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
        video_fps: Optional[float] = None,
    ):
        """
        Fetch video frames for a specific segment.
        
        Args:
            video_path: Path to the video file
            segment: [start, end] for a segment (left-closed, right-open) or [time] for a single frame
            min_frames: Minimum frames for this segment (weighted). Defaults to self.min_frames. Must be >= 1.
            max_frames: Maximum frames for this segment (weighted). Defaults to self.max_frames. Must be >= 1.
            video_fps: Target frames per second for video sampling. If None, uses self.video_fps.
        
        Returns:
            Tuple of (video_tensor, video_metadata)
        """
        # Use provided min/max or fall back to defaults, ensure >= 1
        min_frames = max(1, min_frames if min_frames is not None else self.min_frames)
        max_frames = max(1, max_frames if max_frames is not None else self.max_frames)
        # Use provided video_fps or fall back to self.video_fps
        target_video_fps = video_fps if video_fps is not None else self.video_fps
        
        video_path = clean_video_streams(video_path)
        decoder = VideoDecoder(video_path, num_ffmpeg_threads=0)
        try:
            torchcodec_metadata = decoder.metadata
            
            video_fps = torchcodec_metadata.average_fps
            
            # Calculate duration
            duration = None
            if torchcodec_metadata.end_stream_seconds_from_content is not None and torchcodec_metadata.begin_stream_seconds_from_content is not None:
                duration = torchcodec_metadata.end_stream_seconds_from_content - torchcodec_metadata.begin_stream_seconds_from_content
            if duration is None or duration <= 0:
                duration = torchcodec_metadata.duration_seconds
            
            if len(segment) == 1:
                # Single frame at specified time
                actual_timestamps = self._clamp_timestamps_for_decoder([segment[0]], torchcodec_metadata)
                frame_batch = decoder.get_frames_played_at(actual_timestamps)
                video_tensor = frame_batch.data
                sample_count = 1
            else:
                # Segment [start, end) - left-closed, right-open interval
                start_time, end_time = segment
                segment_duration = end_time - start_time
                
                # Calculate number of frames to sample for this segment
                target_frames = int(math.ceil(segment_duration * target_video_fps))
                target_frames = max(target_frames, min_frames)
                target_frames = min(target_frames, max_frames)
                
                # Generate timestamps for uniform sampling within segment
                if target_frames == 1:
                    actual_timestamps = [start_time]  # Use start_time for single frame
                else:
                    # Sample uniformly within [start, end), endpoint=False for left-closed right-open
                    actual_timestamps = np.linspace(start_time, end_time, target_frames, endpoint=False).tolist()
                
                actual_timestamps = self._clamp_timestamps_for_decoder(actual_timestamps, torchcodec_metadata)

                # Use multithreading for extraction
                result = timestamp_decode_with_multithreading(actual_timestamps, self.num_extract_threads, video_path)
                video_tensor = result["data"]
                sample_count = len(actual_timestamps)
            
            # Create VideoMetadata
            video_metadata = VideoMetadata(
                total_num_frames=sample_count,
                fps=video_fps,
                duration=duration,
                video_backend="torchcodec",
                height=torchcodec_metadata.height,
                width=torchcodec_metadata.width,
                frames_indices=None
            )
            
            # Store actual timestamps as a custom attribute for _calculate_timestamps to use
            video_metadata.actual_timestamps = actual_timestamps
            
            return video_tensor, video_metadata
        finally:
            del decoder

    def fetch_videos(
        self, 
        video_url_or_urls: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]], 
        sample_indices_fn=None, 
        video_fps: Optional[float] = None,
        min_frames: Optional[int] = None,
        max_frames: Optional[int] = None,
    ):
        """
        Override fetch_videos to use torchcodec for frame extraction.
        
        This method uses torchcodec with multithreading for efficient frame extraction.
        Frame count is calculated by the calculate_num_frames method
        (fps-based with min/max constraints).
        
        Args:
            video_url_or_urls: Can be one of:
                - str: Single video path
                - Dict: Video with segments {"video_path": str, "segments": List[List[float]]}
                - List[Union[str, Dict]]: List of video paths or segment dicts
            sample_indices_fn: (Not used) Kept for compatibility with base class signature.
            video_fps: Target frames per second for video sampling. If None, uses self.video_fps.
            min_frames: Minimum number of frames to sample. If None, uses self.min_frames.
            max_frames: Maximum number of frames to sample. If None, uses self.max_frames.
        
        Returns:
            Tuple of (videos, metadata) where videos are torch.Tensors and metadata are VideoMetadata objects.
        """
        # Use provided values or fall back to self defaults
        effective_video_fps = video_fps if video_fps is not None else self.video_fps
        effective_min_frames = min_frames if min_frames is not None else self.min_frames
        effective_max_frames = max_frames if max_frames is not None else self.max_frames
        # Handle recursive calls for lists
        if isinstance(video_url_or_urls, list):
            all_videos = []
            all_metadata = []
            if len(video_url_or_urls) == 1:
                per_video_max_frames = [effective_max_frames]
            else:
                per_video_max_frames = self._allocate_max_frames_for_multiple_videos(
                    video_url_or_urls,
                    effective_max_frames,
                )
            for x, allocated_max_frames in zip(video_url_or_urls, per_video_max_frames):
                result = self.fetch_videos(
                    x, 
                    video_fps=effective_video_fps,
                    min_frames=effective_min_frames,
                    max_frames=allocated_max_frames,
                )
                # Check if result is from segment expansion (returns lists) or single item
                if isinstance(result[0], list):
                    all_videos.extend(result[0])
                    all_metadata.extend(result[1])
                else:
                    all_videos.append(result[0])
                    all_metadata.append(result[1])
            return all_videos, all_metadata
        
        # Handle dict with segments - returns lists (one per segment)
        if isinstance(video_url_or_urls, dict):
            video_path = video_url_or_urls["video_path"]
            segments = video_url_or_urls["segments"]
            
            return self._fetch_video_segments_batched(
                video_path,
                segments,
                min_frames=effective_min_frames,
                max_frames=effective_max_frames,
                video_fps=effective_video_fps,
            )
        
        # Single video path
        video_path = video_url_or_urls
        
        # Clean video streams first (remove extra streams if needed)
        video_path = cached_clean_video_streams(video_path)

        decoder = None
        try:
            # Create VideoDecoder only once for both metadata and frame extraction
            decoder = VideoDecoder(video_path, num_ffmpeg_threads=0)
            torchcodec_metadata = decoder.metadata

            duration = None
            if torchcodec_metadata.end_stream_seconds_from_content is not None and torchcodec_metadata.begin_stream_seconds_from_content is not None:
                duration = torchcodec_metadata.end_stream_seconds_from_content - torchcodec_metadata.begin_stream_seconds_from_content
            
            if duration is None or duration <= 0:
                duration = torchcodec_metadata.duration_seconds
            
            # Use num_frames_from_content for accurate frame count (consistent with extraction)
            total_frames_in_video = torchcodec_metadata.num_frames_from_content
            
            # Create VideoMetadata object for sample_frames method
            temp_metadata = VideoMetadata(
                total_num_frames=total_frames_in_video,
                fps=torchcodec_metadata.average_fps,
                duration=duration,
                video_backend="torchcodec",
                height=torchcodec_metadata.height,
                width=torchcodec_metadata.width,
                frames_indices=None
            )
            
            # Use calculate_num_frames method to get the number of frames to sample
            sample_frames_count = self.calculate_num_frames(
                temp_metadata, 
                fps=effective_video_fps,
                min_frames=effective_min_frames,
                max_frames=effective_max_frames,
            )
            
            # Ensure sample count is valid
            effective_sample_count = min(sample_frames_count, total_frames_in_video)
            if effective_sample_count == 0:
                raise ValueError(f"Cannot extract frames: video has 0 frames or specified frame count is 0")
            
            # Generate uniform frame indices
            frame_indices = np.linspace(0, total_frames_in_video - 1, effective_sample_count).astype(np.int32)
            # Ensure indices are valid and remove duplicates
            frame_indices = np.unique(np.clip(frame_indices, 0, total_frames_in_video - 1))
            
            # Extract frames using multithreading (decoder is created inside each thread for thread safety)
            result = decode_with_multithreading(frame_indices.tolist(), num_threads=self.num_extract_threads, video_path=video_path)
            
            # Extract frame tensor (N, C, H, W)
            frames_tensor = result["data"]
            
            # Create final VideoMetadata object
            video_metadata = VideoMetadata(
                total_num_frames=len(frame_indices),
                fps=torchcodec_metadata.average_fps,
                duration=duration,
                video_backend="torchcodec",
                height=torchcodec_metadata.height,
                width=torchcodec_metadata.width,
                frames_indices=frame_indices
            )
            
            # Ensure frames are in (T, C, H, W) format
            if frames_tensor.dim() == 4:  # (N, C, H, W)
                video_tensor = frames_tensor
            else:
                raise ValueError(f"Unexpected frame tensor shape: {frames_tensor.shape}")
            
            return video_tensor, video_metadata
            
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            traceback.print_exc()
            raise ValueError(f"Failed to load video {video_path}: {e}")
        finally:
            if decoder is not None:
                del decoder

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: Optional[SizeDict] = None,
        interpolation: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        video_max_pixels = getattr(self, "video_max_pixels", None)
        if video_max_pixels is not None:
            total_volume = sum(
                sv.shape[0] * sv.shape[1] * sv.shape[3] * sv.shape[4]
                for sv in grouped_videos.values()
            )
        else:
            total_volume = 0

        for shape, stacked_videos in grouped_videos.items():
            B, T, C, H, W = stacked_videos.shape
            num_frames, height, width = T, H, W
            # Convert to RGB if needed (reuse from base class)
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                if video_max_pixels is not None and total_volume > 0:
                    allocated_max_pixels = int(video_max_pixels * (T * H * W) / total_volume)
                else:
                    allocated_max_pixels = size.longest_edge
                resized_height, resized_width = smart_resize(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=allocated_max_pixels,
                    per_frame_min_pixels=size.shortest_edge,
                    per_frame_max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(B * T, C, H, W)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
                stacked_videos = stacked_videos.view(B, T, C, resized_height, resized_width)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        # Group videos by size for further processing
        # Needed in case do_resize is False, or resize returns videos with different sizes
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            # Fused rescale and normalize
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            # Check that videos have `num_frames` divisible by `temporal_patch_size`
            if patches.shape[1] % temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches_device = patches.device
            # NPU: max 8D tensors — route the 10D permute+reshape through CPU.
            # CUDA handles 10D natively — keep it on-device.
            if patches_device.type == "npu":
                patches = patches.cpu()
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            ).to(patches_device)

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)
        data = {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess(
        self,
        videos: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]],
        **kwargs,
    ) -> BatchFeature:
        """
        Preprocess videos for the model.
        
        This method overrides the base class to handle two video input formats:
        1. String path: "path/to/video.mp4"
        2. Dict with segments: {"video_path": "...", "segment": [[start, end], [time], ...]}
        
        Args:
            videos: Video input(s) in one of the supported formats.
            **kwargs: Additional arguments passed to _preprocess.
            
        Returns:
            BatchFeature with pixel_values_videos, video_grid_thw, and optionally video_metadata.
        """
        # Validate kwargs
        validate_kwargs(
            captured_kwargs=kwargs.keys(),
            valid_processor_keys=list(self.valid_kwargs.__annotations__.keys()) + ["return_tensors"],
        )
        
        # Set default kwargs from self
        for kwarg_name in self.valid_kwargs.__annotations__:
            kwargs.setdefault(kwarg_name, getattr(self, kwarg_name, None))
        
        # Pop kwargs that are handled separately
        return_tensors = kwargs.pop("return_tensors", None)
        return_metadata = kwargs.pop("return_metadata", False)
        input_data_format = kwargs.pop("input_data_format", None)
        device = kwargs.pop("device", None)
        kwargs.pop("video_metadata", None)  # We generate our own metadata
        kwargs.pop("do_sample_frames", None)  # We handle sampling ourselves
        kwargs.pop("data_format", None)  # Not used
        
        # Normalize input to list format
        if not isinstance(videos, list):
            videos = [videos]
        
        # Get video processing params from kwargs (may be passed explicitly for per-batch configuration)
        video_fps = kwargs.pop("video_fps", None)
        min_frames = kwargs.pop("min_frames", None)
        max_frames = kwargs.pop("max_frames", None)
        
        # Use fetch_videos to handle both string and dict formats
        video_tensors, video_metadata = self.fetch_videos(
            videos, 
            video_fps=video_fps,
            min_frames=min_frames,
            max_frames=max_frames,
        )
        
        # Prepare video tensors using _prepare_input_videos
        prepared_videos = self._prepare_input_videos(
            videos=video_tensors,
            input_data_format=input_data_format,
            device=device,
        )
        
        # Process kwargs for _preprocess
        kwargs = self._further_process_kwargs(**kwargs)
        self._validate_preprocess_kwargs(**kwargs)
        
        # Call _preprocess with prepared videos
        result = self._preprocess(videos=prepared_videos, return_tensors=return_tensors, **kwargs)
        
        # Add metadata if requested
        if return_metadata:
            result["video_metadata"] = video_metadata
        
        return result


__all__ = ["MossVLVideoProcessor"]

