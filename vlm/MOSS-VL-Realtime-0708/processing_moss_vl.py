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
"""
Processor class for Moss-VL.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, SizeDict
from transformers.image_processing_utils_fast import group_images_by_shape, reorder_images
from transformers.utils import TensorType
from transformers.processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
    VideosKwargs,
)
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


logger = logging.get_logger(__name__)


class MossVLImageProcessorFast(Qwen2VLImageProcessorFast):
    """
    Custom image processor that overrides _preprocess to support multi_image_max_pixels.
    Inherits from Qwen2VLImageProcessorFast.
    """
    # Multi-image batch total pixels limit (read from config)
    multi_image_max_pixels = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, 'interpolation') or self.interpolation is None:
            self.interpolation = "BICUBIC"

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        interpolation: Optional["F.InterpolationMode"] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, list[float]]] = None,
        image_std: Optional[Union[float, list[float]]] = None,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        merge_size: int = 2,
        disable_grouping: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ):
        """Override _preprocess to use custom smart_resize with batch-level max_pixels.
        
        multi_image_max_pixels is treated as a batch-level total budget, proportionally allocated
        to each image based on its original pixel count. min_pixels remains a per-image
        constraint. multi_image_max_pixels can be configured separately from longest_edge.
        """
        min_pixels = size["shortest_edge"]
        max_pixels = size["longest_edge"]  # Per-image upper limit
        # Use multi_image_max_pixels if configured, otherwise fall back to longest_edge
        multi_image_max_pixels = getattr(self, "multi_image_max_pixels", None) or max_pixels
        
        # Calculate total original pixels across all images in the batch
        # This is used to proportionally allocate max_pixels to each image
        total_original_pixels = sum(img.shape[-2] * img.shape[-1] for img in images)

        # Group images by size for batched resizing
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            height, width = stacked_images.shape[-2:]
            if do_resize:
                # Calculate proportional max_pixels for images with this shape
                # Each image's max_pixels is allocated based on its proportion of total pixels
                original_pixels = height * width
                if total_original_pixels > 0:
                    proportion = original_pixels / total_original_pixels
                    proportional_max_pixels = int(multi_image_max_pixels * proportion)
                else:
                    proportional_max_pixels = multi_image_max_pixels
                
                # Ensure proportional max_pixels is within [min_pixels, max_pixels] range
                # min_pixels: per-image lower limit (shortest_edge)
                # max_pixels: per-image upper limit (longest_edge)
                proportional_max_pixels = max(proportional_max_pixels, min_pixels)
                proportional_max_pixels = min(proportional_max_pixels, max_pixels)
                
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=min_pixels,
                    max_pixels=proportional_max_pixels,
                )
                stacked_images = self.resize(
                    image=stacked_images,
                    size=SizeDict(height=resized_height, width=resized_width),
                    interpolation=interpolation,
                )
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)
        
        # Warn if multi-image batch exceeds multi_image_max_pixels due to min_pixels constraint
        if len(images) > 1:
            total_resized_pixels = sum(img.shape[-2] * img.shape[-1] for img in resized_images)
            if total_resized_pixels > multi_image_max_pixels:
                logger.warning_once(
                    f"Multi-image batch total pixels ({total_resized_pixels}) exceeds multi_image_max_pixels ({multi_image_max_pixels}). "
                    f"This may happen when image_count * min_pixels > multi_image_max_pixels."
                )

        # Group images by size for further processing
        # Needed in case do_resize is False, or resize returns images with different sizes
        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        processed_grids = {}
        for shape, stacked_images in grouped_images.items():
            resized_height, resized_width = stacked_images.shape[-2:]
            # Fused rescale and normalize
            patches = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            if patches.ndim == 4:
                # add a temporal dimension if we have images
                patches = patches.unsqueeze(1)
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
            # Reorder dimensions to group grid and patch information for subsequent flattening.
            # (batch, grid_t, grid_h, grid_w, merge_h, merge_w, channel, temp_patch_size, patch_h, patch_w)
            # NPU supports max 8D tensors; route the 10D permute+reshape through
            # CPU there. CUDA handles 10D natively — keep it on-device.
            patches_device = patches.device
            if patches_device.type == "npu":
                patches = patches.cpu()
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            ).to(patches_device)

            processed_images_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_grids = reorder_images(processed_grids, grouped_images_index)
        pixel_values = torch.cat(processed_images, dim=0)
        image_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}, tensor_type=return_tensors
        )

def _to_numpy(x):
    """
    Convert various tensor types to numpy array.
    Supports torch.Tensor, tf.Tensor, jax.Array, np.ndarray, lists, and primitives.
    
    Args:
        x: Input value that can be a tensor from various frameworks or a Python primitive
        
    Returns:
        np.ndarray: NumPy array representation of the input
    """
    # Already numpy
    if isinstance(x, np.ndarray):
        return x
    
    # Torch tensor or TensorFlow tensor (both have .numpy() method)
    if hasattr(x, 'numpy'):
        # For torch tensors on CUDA, need to move to CPU first
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        # For TensorFlow or already on CPU
        return x.numpy()
    
    # JAX arrays and other array-like objects that support __array__ protocol
    if hasattr(x, '__array__'):
        return np.asarray(x)
    
    # Python primitives (list, tuple, int, float)
    return np.array(x)


def _split_array_or_tensor(x, split_indices):
    """Split along the first dimension while preserving tensor/array type."""
    split_indices = [int(idx) for idx in split_indices]
    if isinstance(x, torch.Tensor):
        if not split_indices:
            return [x]
        chunks = []
        start = 0
        for end in split_indices:
            chunks.append(x[start:end])
            start = end
        chunks.append(x[start:])
        return chunks
    return np.split(x, split_indices)


def _concat_array_or_tensor(items, axis=0):
    """Concatenate while preserving tensor/array type and device."""
    if not items:
        return None

    if any(isinstance(item, torch.Tensor) for item in items):
        ref = next(item for item in items if isinstance(item, torch.Tensor))
        tensor_items = [
            item
            if isinstance(item, torch.Tensor)
            else torch.as_tensor(item, device=ref.device, dtype=ref.dtype)
            for item in items
        ]
        return torch.cat(tensor_items, dim=axis)

    return np.concatenate(items, axis=axis)


def _stack_array_or_tensor(items, axis=0):
    """Stack while preserving tensor/array type and device."""
    if not items:
        return None

    if any(isinstance(item, torch.Tensor) for item in items):
        ref = next(item for item in items if isinstance(item, torch.Tensor))
        tensor_items = [
            item
            if isinstance(item, torch.Tensor)
            else torch.as_tensor(item, device=ref.device, dtype=ref.dtype)
            for item in items
        ]
        return torch.stack(tensor_items, dim=axis)

    return np.stack(items, axis=axis)


class MossVLImagesKwargs(ImagesKwargs):
    min_pixels: Optional[int]
    max_pixels: Optional[int]
    patch_size: Optional[int]
    temporal_patch_size: Optional[int]
    merge_size: Optional[int]
    interpolation: Optional[str]


class MossVLVideosKwargs(VideosKwargs, total=False):
    video_fps: Optional[Union[int, float]]
    min_frames: Optional[int]
    max_frames: Optional[int]
    num_extract_threads: Optional[int]


class MossVLProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: MossVLImagesKwargs
    videos_kwargs: MossVLVideosKwargs
    # _defaults = {
    #     "text_kwargs": {
    #         "padding": True,                    # 👈 启用 padding
    #         "padding_side": "left",            # 👈 左 padding
    #         "pad_to_multiple_of": 8,           # 👈 pad 到 8 的倍数
    #         "return_token_type_ids": False,
    #         "return_mm_token_type_ids": False,
    #     },
    #     "videos_kwargs": {"return_metadata": True},
    # }
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }

class MossVLProcessor(ProcessorMixin):
    r"""
    Constructs a Moss-VL processor which wraps a Qwen2VL image processor, Moss-VL video processor and a Qwen2 tokenizer
    into a single processor.

    [`MossVLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`], [`MossVLVideoProcessor`] and [`Qwen2TokenizerFast`].
    See the [`~MossVLProcessor.__call__`] and [`~MossVLProcessor.decode`] for more information.

    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`MossVLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer", "video_processor"]
    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None, 
        video_processor=None, 
        chat_template=None,
        **kwargs
    ):
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        

        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token

        
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        
        self.vision_start_token = (
            "<|vision_start|>" if not hasattr(tokenizer, "vision_start_token") else tokenizer.vision_start_token
        )
        self.vision_end_token = (
            "<|vision_end|>" if not hasattr(tokenizer, "vision_end_token") else tokenizer.vision_end_token
        )

        # Placeholders used in input text
        self.image_placeholder = "<|image|>"
        self.video_placeholder = "<|video|>"

        self.time_start_token = "<|time_start|>"
        self.time_end_token = "<|time_end|>"
        
        # EOS token for labels generation (assistant's response should end with this)
        self.im_end_token = "<|im_end|>"
        self.im_end_token_id = tokenizer.convert_tokens_to_ids(self.im_end_token)
        
        # Vision-related token ids (all should be masked in labels)
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.vision_start_token)
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.vision_end_token)
        
        # Token ids that should always be masked in labels (e.g. <|image_pad|>)
        self.mask_token_ids = {self.image_token_id}

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]] = None,
        labels_spans: Optional[Union[List[tuple], List[List[tuple]]]] = None,
        ignore_index: int = -100,
        **kwargs: Unpack[MossVLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s)/video(s).

        Args:
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded.
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared.
            videos (`str`, `Dict`, `list[str]`, `list[Dict]`):
                The video or batch of videos to be prepared. Each video can be:
                - A string path to a video file
                - A dict with keys:
                    - "video_path": str, path to the video file
                    - "segments": list of segments, where each segment is:
                        - [start, end]: a time segment (left-closed, right-open interval in seconds)
                        - [time]: a single frame at the specified time (in seconds)
                  The number of segments should match the number of video placeholders in the text.
            labels_spans (`list[list[int]]`, `list[list[list[int]]]`, *optional*):
                Character-level spans indicating assistant regions in original text.
                Each span is a [start, end] list with inclusive start and exclusive end.
                Example: [[10, 50], [100, 150]] means characters [10:50) and [100:150) are assistant.
                Note: Use list (not tuple) for spans as they will be modified in place during processing.
                When provided, the processor will generate `labels` in the output, where:
                - Non-assistant tokens have value `ignore_index` (-100 by default)
                - Image tokens always have value `ignore_index` even in assistant part
                - Other assistant tokens have their token id as label
            ignore_index (`int`, *optional*, defaults to -100):
                Value for masked positions in labels.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.


        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **pixel_values** -- Pixel values to be fed to a model (concatenation of images and videos).
            - **grid_thw** -- List of grid sizes (t, h, w) for each media item.
            - **media_nums_per_sample** -- List of number of media items per sample.
            - **labels** -- (Optional) Labels for training, only present when `labels_spans` is provided.
        """
        # Merge kwargs with defaults
        output_kwargs = self._merge_kwargs(
            MossVLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        
        # Step 1: Process images if provided
        if images is not None:
            images_kwargs = output_kwargs["images_kwargs"].copy()
            images_kwargs["return_tensors"] = None
            image_inputs = self.image_processor(images=images, **images_kwargs)
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        # Step 2: Process videos if provided
        if videos is not None:
            videos_kwargs = output_kwargs["videos_kwargs"].copy()
            videos_kwargs["return_tensors"] = None
            videos_inputs = self.video_processor(videos=videos, **videos_kwargs)
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
        else:
            videos_inputs = {}
            video_grid_thw = None
            video_metadata = None

        # Step 3: Process text with placeholder replacement
        if text is None or (isinstance(text, str) and len(text.strip()) == 0):
            raise ValueError("Text input is required for MossVL processor and cannot be empty.")
        
        if not isinstance(text, list):
            text = [text]
        
        text = text.copy()  # Copy to avoid in-place modifications
        
        # Prepare labels_spans if provided
        # labels_spans format: List[List[List[int]]] - batch of samples, each sample has multiple spans
        # Each span is [start, end] (list, not tuple) so it can be modified in place
        should_create_labels = labels_spans is not None
        if should_create_labels:
            # Ensure batch format: convert single sample spans to batch format
            # Single sample: [[start, end], [start, end], ...]
            # Batch: [[[start, end], ...], [[start, end], ...], ...]
            if labels_spans and isinstance(labels_spans[0], list) and len(labels_spans[0]) == 2 and isinstance(labels_spans[0][0], int):
                labels_spans = [labels_spans]

        # Step 3.0-pre: Check if we need to reorder (when both images and videos exist)
        # If only one media type exists, we can skip the expensive split+reorder+concat
        has_images = images is not None and "pixel_values" in image_inputs
        has_videos = videos is not None and "pixel_values_videos" in videos_inputs
        needs_reorder = has_images and has_videos
        
        image_pixel_values_list = []
        video_pixel_values_list = []
        
        # Step 3.0: Record the order of media in original text (before replacement)
        # This will be used later to correctly order pixel_values and grid_thw
        media_order_per_sample = []
        for i in range(len(text)):
            media_order = []
            temp_text = text[i]
            pos = 0
            while pos < len(temp_text):
                img_pos = temp_text.find(self.image_placeholder, pos)
                vid_pos = temp_text.find(self.video_placeholder, pos)
                
                if img_pos == -1 and vid_pos == -1:
                    break
                
                if img_pos != -1 and (vid_pos == -1 or img_pos < vid_pos):
                    media_order.append(("image", img_pos))
                    pos = img_pos + len(self.image_placeholder)
                elif vid_pos != -1:
                    media_order.append(("video", vid_pos))
                    pos = vid_pos + len(self.video_placeholder)
            
            media_order_per_sample.append(media_order)
        
        # Step 3.0.1: Check if any sample has no media (empty samples need blank image)
        # If there are empty samples, we need to enter slow path to handle them properly
        has_empty_samples = any(len(order) == 0 for order in media_order_per_sample)
        if has_empty_samples:
            needs_reorder = True
        
        # Split pixel values for reordering if needed
        if needs_reorder:
            if has_images:
                flat_pixel_values = image_inputs["pixel_values"]
                flat_grid_thw = image_inputs["image_grid_thw"]
                # grid_thw is (t, h, w), num_patches = t * h * w
                patch_counts = [int(np.prod(_to_numpy(grid))) for grid in flat_grid_thw]
                if len(patch_counts) == 1:
                    # Single image case: no need to split
                    image_pixel_values_list = [flat_pixel_values]
                elif len(patch_counts) > 1:
                    # Multiple images: split by cumulative counts
                    split_indices = np.cumsum(patch_counts)[:-1]
                    image_pixel_values_list = _split_array_or_tensor(
                        flat_pixel_values, split_indices
                    )

            if has_videos:
                flat_video_values = videos_inputs["pixel_values_videos"]
                flat_video_grid = videos_inputs["video_grid_thw"]
                video_patch_counts = [int(np.prod(_to_numpy(grid))) for grid in flat_video_grid]
                if len(video_patch_counts) == 1:
                    # Single video case: no need to split
                    video_pixel_values_list = [flat_video_values]
                elif len(video_patch_counts) > 1:
                    # Multiple videos: split by cumulative counts
                    split_indices = np.cumsum(video_patch_counts)[:-1]
                    video_pixel_values_list = _split_array_or_tensor(
                        flat_video_values, split_indices
                    )
        
        # Step 3.1: Replace placeholders (simple replacement, no expansion yet)
        # In MossVL, one image placeholder = one image token
        # One video placeholder = one video token (will be expanded later)
        for i in range(len(text)):
            if should_create_labels:
                # Replace and update spans for image placeholders
                text[i], labels_spans[i] = self._replace_and_update_spans(
                    text[i], self.image_placeholder, self.image_token, labels_spans[i]
                )
                # Replace and update spans for video placeholders
                text[i], labels_spans[i] = self._replace_and_update_spans(
                    text[i], self.video_placeholder, self.video_token, labels_spans[i]
                )
            else:
                text[i] = text[i].replace(self.image_placeholder, self.image_token)
                text[i] = text[i].replace(self.video_placeholder, self.video_token)
        
        # Step 3.2: Validate token counts 
        n_images_in_text = [t.count(self.image_token) for t in text]
        n_videos_in_text = [t.count(self.video_token) for t in text]
        
        # Count placeholders in text
        total_images_in_text = sum(n_images_in_text)
        total_videos_in_text = sum(n_videos_in_text)
        
        # Count actual images and videos provided
        total_images_provided = len(image_grid_thw) if image_grid_thw is not None else 0
        total_videos_provided = len(video_grid_thw) if video_grid_thw is not None else 0
        
        # Validate image counts
        if total_images_in_text != total_images_provided:
            raise ValueError(
                "Number of image tokens does not match number of images provided. "
                f"Found {total_images_in_text} image tokens in text and {total_images_provided} images."
            )
        
        # Validate video counts
        if total_videos_in_text != total_videos_provided:
            raise ValueError(
                "Number of video tokens does not match number of videos provided. "
                f"Found {total_videos_in_text} video tokens in text and {total_videos_provided} videos."
            )
        
        # Step 3.3: Expand video tokens with timestamps
        # Now expand each video token to multiple tokens (one per frame) with timestamps
        if video_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "MossVL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    # Calculate timestamps
                    # Use actual_timestamps if available (for segments), otherwise use frames_indices
                    actual_timestamps = getattr(metadata, 'actual_timestamps', None)
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.total_num_frames,
                        metadata.fps,
                        metadata.duration,
                        self.video_processor.temporal_patch_size,
                        actual_timestamps=actual_timestamps,
                    )

                    # Build video placeholder: one video token per frame with timestamp
                    # video_grid_thw[index][0] is the temporal dimension (number of frames after merging)

                    video_tokens = []
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        # Format: <|time_start|>X.X seconds<|time_end|><|image_pad|>
                        video_tokens.append(
                            f"{self.time_start_token}{curr_time:.1f} seconds{self.time_end_token}{self.image_token}"
                        )
                    
                    # Wrap the entire video sequence with vision_start and vision_end tokens
                    video_placeholder = f"{self.vision_start_token}{''.join(video_tokens)}{self.vision_end_token}"
                    
                    # Replace the video token with expanded sequence and update spans if needed
                    if should_create_labels:
                        text[i], labels_spans[i] = self._replace_and_update_spans(
                            text[i], self.video_token, video_placeholder, labels_spans[i], replace_count=1
                        )
                    else:
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
                    index += 1
                


        # Step 4: Tokenize text
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        
        # Request offset_mapping if we need to create labels
        if should_create_labels:
            output_kwargs["text_kwargs"]["return_offsets_mapping"] = True
        
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        
        # ignore check_special_mm_tokens nums in test and input ids.
        # self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])
        
        # Create labels if labels_spans was provided
        if should_create_labels:
            offset_mapping = text_inputs.pop("offset_mapping")
            labels = self._create_labels_from_spans(
                text_inputs["input_ids"],
                offset_mapping,
                labels_spans,
                ignore_index
            )

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        # Step 5: Concatenate pixel_values and grid_thw in sequence order
        # Prepare output
        output_data = {**text_inputs}
        
        if not needs_reorder:
            # Fast path: only one media type, no reordering needed
            final_pixel_values = []
            final_grid_thw = []
            
            if has_images:
                final_pixel_values.append(image_inputs["pixel_values"])
                final_grid_thw.extend(image_grid_thw)
            
            if has_videos:
                final_pixel_values.append(videos_inputs["pixel_values_videos"])
                final_grid_thw.extend(video_grid_thw)
            
            if final_pixel_values:
                output_data["pixel_values"] = np.concatenate(final_pixel_values, axis=0) if len(final_pixel_values) > 1 else final_pixel_values[0]
            
            if final_grid_thw:
                output_data["grid_thw"] = np.stack(final_grid_thw, axis=0)
            
            # Calculate media_nums_per_sample
            media_nums_per_sample = []
            for batch_idx in range(len(text)):
                media_order = media_order_per_sample[batch_idx]
                media_nums_per_sample.append(len(media_order) if len(media_order) > 0 else 1)
            
            # Don't add media_nums_per_sample to output_data yet
            # Will add it after BatchFeature to keep it as list
            
        else:
            # Slow path: both images and videos exist, need reordering
            final_pixel_values = []
            final_grid_thw = []
            media_nums_per_sample = []
            
            # Global indices to track position in flattened image/video arrays
            global_image_idx = 0
            global_video_idx = 0
            
            for batch_idx in range(len(text)):
                # Use the recorded media order from Step 3.0
                media_order = media_order_per_sample[batch_idx]
                
                if len(media_order) == 0:
                    # If no media provided for this sample, add a blank image
                    media_nums_per_sample.append(1)
                    min_pixels = 128 * 128
                    patch_size = getattr(self.image_processor, "patch_size", None) or 16
                    temporal_patch_size = getattr(self.image_processor, "temporal_patch_size", None) or 1
                    merge_size = getattr(self.image_processor, "merge_size", None) or 2
                    
                    factor = patch_size * merge_size
                    side = int(np.ceil(np.sqrt(min_pixels) / factor) * factor)
                    grid_h = side // patch_size
                    grid_w = side // patch_size
                    grid_t = 1
                    
                    # Channel = 3 (RGB)
                    channel = 3
                    dim = channel * temporal_patch_size * patch_size * patch_size
                    num_patches = grid_t * grid_h * grid_w
                    
                    blank_pixel_values = np.zeros((num_patches, dim), dtype=np.float32)
                    blank_grid_thw = np.array([grid_t, grid_h, grid_w], dtype=np.int64)
                    
                    final_pixel_values.append(blank_pixel_values)
                    final_grid_thw.append(blank_grid_thw)
                else:
                    media_nums_per_sample.append(len(media_order))
                    
                    # Collect media data according to the recorded order
                    for media_type, _ in media_order:
                        if media_type == "image" and image_grid_thw is not None:
                            # Get image data
                            if image_pixel_values_list:
                                final_pixel_values.append(image_pixel_values_list[global_image_idx])
                            final_grid_thw.append(image_grid_thw[global_image_idx])
                            global_image_idx += 1
                        elif media_type == "video" and video_grid_thw is not None:
                            # Get video data
                            if video_pixel_values_list:
                                final_pixel_values.append(video_pixel_values_list[global_video_idx])
                            final_grid_thw.append(video_grid_thw[global_video_idx])
                            global_video_idx += 1
            
            # Concatenate/stack to unified format
            if final_pixel_values:
                output_data["pixel_values"] = _concat_array_or_tensor(
                    final_pixel_values, axis=0
                )
            
            if final_grid_thw:
                output_data["grid_thw"] = _stack_array_or_tensor(
                    final_grid_thw, axis=0
                )
            
            # Don't add media_nums_per_sample to output_data yet
            # Will add it after BatchFeature to keep it as list

        # Create cross_attention_mask using media_nums_per_sample
        if "input_ids" in output_data and "grid_thw" in output_data and media_nums_per_sample:
            cross_attention_mask = self._create_cross_attention_mask(
                output_data["input_ids"],
                output_data["grid_thw"],
                media_nums_per_sample,
                output_data.get("attention_mask", None)
            )
            output_data["cross_attention_mask"] = cross_attention_mask
        
        # Add labels to output if created
        if should_create_labels:
            output_data["labels"] = labels

        # BatchFeature will handle conversion to pt/tf/jax/np based on tensor_type
        batch_feature = BatchFeature(data=output_data, tensor_type=return_tensors)
        
        # Add media_nums_per_sample after BatchFeature to keep it as list (not tensor)
        if media_nums_per_sample:
            batch_feature["media_nums_per_sample"] = media_nums_per_sample
        
        return batch_feature

    def _create_cross_attention_mask(self, input_ids, grid_thw, media_nums_per_sample, attention_mask=None):
        """
        Create cross_attention_mask of shape (batch_size, 1, text_len, num_images).
        Video frames are treated as individual images.
        Mask values: True for masked, False for visible.
        Causal masking: text can see images that appear at or before the text position.
        
        Args:
            input_ids: List of token ids
            grid_thw: Grid sizes for each media item
            media_nums_per_sample: Number of media items per sample
            attention_mask: Optional attention mask to filter out padding positions
        """
        batch_size = len(input_ids)
        max_text_len = max(len(ids) for ids in input_ids)
        
        # Calculate total frames per sample to find max_num_frames
        total_frames_per_sample = []
        media_idx = 0
        for b in range(batch_size):
            num_media = media_nums_per_sample[b]
            if num_media == 0:
                total_frames_per_sample.append(0)
                continue
                
            sample_frames = 0
            for _ in range(num_media):
                # grid_thw is (N, 3) where first dim is t (num_frames)
                t = grid_thw[media_idx][0]
                if isinstance(t, torch.Tensor):
                    t = int(t.item())
                else:
                    t = int(t)
                sample_frames += t
                media_idx += 1
            total_frames_per_sample.append(sample_frames)
            
        max_num_frames = max(total_frames_per_sample) if total_frames_per_sample else 0
        
        if max_num_frames == 0:
            return None
            
        # Vectorized implementation for speed
        
        # 1. Pad input_ids to create a tensor
        # We use -1 as pad value since token ids are positive
        input_ids_tensor = torch.full((batch_size, max_text_len), -1, dtype=torch.long)
        for b, ids in enumerate(input_ids):
            l = len(ids)
            input_ids_tensor[b, :l] = torch.tensor(ids, dtype=torch.long)
            
        # 2. Identify image tokens
        is_image_token = (input_ids_tensor == self.image_token_id)
        
        # 3. Compute cumulative image tokens (how many image tokens appeared up to position t)
        # shape: (batch_size, text_len)
        cum_image_tokens = is_image_token.cumsum(dim=1)
        
        # 4. Create frame indices
        # shape: (1, 1, max_num_frames)
        frame_indices = torch.arange(max_num_frames).reshape(1, 1, -1)
        
        # 5. Determine visibility based on causal relationship
        # Text at `t` sees frame `i` if `cum_image_tokens[t] > i`
        # Because if frame `i` is the (i+1)-th image token, it becomes visible when count reaches i+1
        # shape: (batch_size, text_len, max_num_frames)
        visible_mask = cum_image_tokens.unsqueeze(-1) > frame_indices
        
        # 6. Apply attention_mask if provided
        if attention_mask is not None:
            # Convert to tensor if needed
            if isinstance(attention_mask, torch.Tensor):
                 attn_mask_tensor = attention_mask
            else:
                 # List of lists
                 attn_mask_tensor = torch.zeros((batch_size, max_text_len), dtype=torch.long)
                 for b, mask_row in enumerate(attention_mask):
                     l = len(mask_row)
                     attn_mask_tensor[b, :l] = torch.tensor(mask_row, dtype=torch.long)
            
            # shape: (batch_size, text_len, 1)
            valid_text = (attn_mask_tensor.unsqueeze(-1) == 1)
            visible_mask = visible_mask & valid_text
            
        # 7. Mask out frames that don't exist for a sample
        # shape: (batch_size, 1, 1)
        total_frames_tensor = torch.tensor(total_frames_per_sample).reshape(batch_size, 1, 1)
        # shape: (batch_size, 1, max_num_frames)
        valid_frames = frame_indices < total_frames_tensor
        
        visible_mask = visible_mask & valid_frames
        
        # 8. Create final mask (True for masked, False for visible)
        mask = ~visible_mask
        
        # 9. Add channel dimension: (batch_size, 1, text_len, max_num_frames)
        mask = mask.unsqueeze(1)
        
        return mask

    def _replace_and_update_spans(
        self,
        text: str,
        old_str: str,
        new_str: str,
        spans: List[List[int]],
        replace_count: int = -1
    ) -> tuple:
        """
        Replace occurrences of old_str with new_str and update spans accordingly.
        
        Args:
            text: The text to perform replacement on
            old_str: String to be replaced
            new_str: String to replace with
            spans: List of [start, end] spans to update (modified in place)
            replace_count: Maximum number of replacements (-1 for all)
            
        Returns:
            Tuple of (new_text, updated_spans)
        """
        delta = len(new_str) - len(old_str)
        result_text = text
        count = 0
        search_start = 0
        
        while True:
            pos = result_text.find(old_str, search_start)
            if pos == -1:
                break
            if replace_count != -1 and count >= replace_count:
                break
            
            # Update all spans that come after this position
            for span in spans:
                if span[0] > pos:
                    # Span starts after replacement point
                    span[0] += delta
                    span[1] += delta
                elif span[1] > pos:
                    # Span ends after replacement point (spans the replacement)
                    span[1] += delta
            
            # Perform the replacement
            result_text = result_text[:pos] + new_str + result_text[pos + len(old_str):]
            search_start = pos + len(new_str)
            count += 1
        
        return result_text, spans

    def _create_labels_from_spans(
        self,
        input_ids: List[List[int]],
        offset_mapping: List[List[tuple]],
        labels_spans: List[List[List[int]]],
        ignore_index: int = -100,
        mask_token_ids: Optional[set] = None
    ) -> List[List[int]]:
        """
        Create labels from spans and offset_mapping.
        
        Args:
            input_ids: Tokenized input ids
            offset_mapping: Character offsets for each token from tokenizer (special tokens included)
            labels_spans: Updated spans indicating assistant regions (after text transformations)
            ignore_index: Value for masked positions
            mask_token_ids: Set of token ids that should always be masked (set to ignore_index)
                in labels, regardless of whether they fall inside a span.
                Defaults to self.mask_token_ids if not provided.
            
        Returns:
            labels: List of label ids, same shape as input_ids
        
        Note:
            - Tokenizer's offset_mapping already includes correct offsets for special tokens in text
            - Only need to mask tokens inside <|vision_start|>...<|vision_end|>
            - Tokens whose id is in mask_token_ids are always masked
            - All other tokens in spans (including special tokens like <|im_end|>) get labels
        """
        if mask_token_ids is None:
            mask_token_ids = self.mask_token_ids
        
        batch_labels = []
        
        for batch_idx in range(len(input_ids)):
            ids = input_ids[batch_idx]
            offsets = offset_mapping[batch_idx]
            spans = labels_spans[batch_idx]
            
            labels = [ignore_index] * len(ids)
            
            # Process each span: find token range and set labels
            for span_start, span_end in spans:
                in_vision = False
                
                # Find tokens that overlap with this span
                for token_idx, (token_id, (char_start, char_end)) in enumerate(zip(ids, offsets)):
                    # Skip tokens completely before this span
                    if char_end <= span_start:
                        continue
                    # Stop when tokens are completely after this span
                    if char_start >= span_end:
                        break
                    
                    # Token overlaps with span, process it
                    # Track vision region: <|vision_start|> ... <|vision_end|>
                    if token_id == self.vision_start_token_id:
                        in_vision = True
                        continue
                    if token_id == self.vision_end_token_id:
                        in_vision = False
                        continue
                    
                    # Skip tokens inside vision region
                    if in_vision:
                        continue
                    
                    # Always mask special tokens that should never have labels
                    if token_id in mask_token_ids:
                        continue
                    
                    # Set label for this token
                    labels[token_idx] = token_id
            
            batch_labels.append(labels)
        
        return batch_labels

    def _calculate_timestamps(
        self, 
        frames_indices: Optional[Union[List[int], np.ndarray]], 
        total_num_frames: int, 
        video_fps: float, 
        duration: float, 
        merge_size: int = 1,
        actual_timestamps: Optional[List[float]] = None
    ):
        """
        Calculate timestamps for video frames.
        
        Args:
            frames_indices: Actual frame indices extracted (if available)
            total_num_frames: Total number of sampled frames
            video_fps: Video frames per second
            duration: Video duration in seconds
            merge_size: Temporal merge size
            actual_timestamps: Pre-calculated actual timestamps (for segments)
            
        Returns:
            List of timestamps (one per merged temporal patch)
        """
        # If actual timestamps are provided (from segment), use them directly
        if actual_timestamps is not None:
            timestamps = list(actual_timestamps)
            
            # Pad timestamps to be multiple of merge_size
            if len(timestamps) % merge_size != 0:
                timestamps.extend([timestamps[-1]] * (merge_size - len(timestamps) % merge_size))
            
            # Frames are merged by merge_size, so we average the timestamps within each temporal patch
            timestamps = [
                (timestamps[i] + timestamps[i + merge_size - 1]) / 2 
                for i in range(0, len(timestamps), merge_size)
            ]
            return timestamps
        
        # Use frames_indices if available, otherwise generate uniformly sampled indices
        if frames_indices is not None:
            if isinstance(frames_indices, np.ndarray):
                indices = frames_indices.tolist()
            else:
                indices = list(frames_indices)
        else:
            # Generate uniformly sampled frame indices
            if total_num_frames <= 1:
                indices = [0]
            else:
                # Uniformly sample frames across the video duration
                indices = np.linspace(0, duration * video_fps - 1, total_num_frames).astype(np.int32).tolist()
        
        # Pad indices to be multiple of merge_size
        if len(indices) % merge_size != 0:
            indices.extend([indices[-1]] * (merge_size - len(indices) % merge_size))
        
        # Convert frame indices to timestamps
        timestamps = [idx / video_fps for idx in indices]
        
        # Frames are merged by merge_size, so we average the timestamps within each temporal patch
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2 
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's batch_decode. 
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's decode.
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor 
                of shape `(batch_size, sequence_length)` or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode` method.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["MossVLProcessor", "MossVLImageProcessorFast"]
