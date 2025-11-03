#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
import folder_paths
import comfy.utils

import numpy as np
import torch.nn.functional as F
import gc
import subprocess
import sys
import datetime
import json
import re

from einops import rearrange
from huggingface_hub import snapshot_download
from .src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from .src.models.TCDecoder import build_tcdecoder
from .src.models.utils import clean_vram, Buffer_LQ4x_Proj
from .src.models import wan_video_dit

def get_device_list():
    devs = ["auto"]
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except Exception:
        pass
    return devs

device_choices = get_device_list()

def log(message:str, message_type:str='normal'):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

def get_ffmpeg_path():
    """Get ffmpeg executable path"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except:
        pass
    
    # Check if ffmpeg is in PATH
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    
    # Check in current directory
    if os.path.isfile("ffmpeg"):
        return os.path.abspath("ffmpeg")
    if os.path.isfile("ffmpeg.exe"):
        return os.path.abspath("ffmpeg.exe")
    
    return None

def tensor_to_bytes(tensor):
    """Convert tensor [0,1] to uint8 numpy array [0,255]"""
    # Ensure tensor is in [0, 1] range
    tensor = torch.clamp(tensor, 0.0, 1.0)
    # Convert to numpy and scale to [0, 255]
    # Use direct multiplication and type conversion for consistency with standard image processing
    arr = tensor.cpu().numpy()
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(folder_paths.models_dir, "FlashVSR")
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")

    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    # Directly scale to target dimensions instead of scaling then cropping
    # This avoids the "zoom in" effect from center cropping
    upscaled_tensor = F.interpolate(tensor_bchw, size=(tH, tW), mode='bicubic', align_corners=False)

    return upscaled_tensor.squeeze(0)

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to('cpu').to(dtype)
        frames.append(tensor_out)

    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    
    del vid_stacked
    clean_vram()
    
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
                
            coords.append((x1, y1, x2, y2))
            
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def init_pipeline(mode, device, dtype):
    model_downlod()
    model_path = os.path.join(folder_paths.models_dir, "FlashVSR")
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist!\nPlease save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!\nPlease save it to "{model_path}"')
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist!\nPlease save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!\nPlease save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!\nPlease save it to "{model_path}"')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()
        
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    
    return pipe

class cqdm:
    def __init__(self, iterable=None, total=None, desc="Processing"):
        self.desc = desc
        self.pbar = None
        self.iterable = None
        self.total = total
        
        if iterable is not None:
            try:
                self.total = len(iterable)
                self.iterable = iter(iterable)
            except TypeError:
                if self.total is None:
                    raise ValueError("Total must be provided for iterables with no length.")

        elif self.total is not None:
            pass
            
        else:
            raise ValueError("Either iterable or total must be provided.")
            
    def __iter__(self):
        if self.iterable is None:
            raise TypeError(f"'{type(self).__name__}' object is not iterable. Did you mean to use it with a 'with' statement?")
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self
    
    def __next__(self):
        if self.iterable is None:
            raise TypeError("Cannot call __next__ on a non-iterable cqdm object.")
        try:
            val = next(self.iterable)
            if self.pbar:
                self.pbar.update(1)
            return val
        except StopIteration:
            raise
            
    def __enter__(self):
        if self.pbar is None:
            self.pbar = comfy.utils.ProgressBar(self.total)
        return self.pbar
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __len__(self):
        return self.total

class FlashVSRNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Sequential video frames as IMAGE tensor batch"
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Using "tiny-long" mode can significantly reduce VRAM used with long video input.'
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                }),
                "target_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 4,
                    "tooltip": "Target output height. 0 for auto (input height * scale)."
                }),
                "target_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 4,
                    "tooltip": "Target output width. 0 for auto (input width * scale)."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use wavelet transform to correct output video color."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable tiling: faster decode but higher VRAM usage.\nSet to True for lower memory consumption at the cost of speed."
                }),
                "tiled_dit": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Significantly reduces VRAM usage at the cost of speed."
                }),
                "tile_size": ("INT", {
                    "default": 256,
                    "min": 32,
                    "max": 1024,
                    "step": 32,
                }),
                "tile_overlap": ("INT", {
                    "default": 24,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload DiT before decoding to reduce VRAM peak at the cost of speed."
                }),
                "sparse_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.5 or 2.0\n1.5 → faster; 2.0 → more stable"
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.0 to 3.0\n1.0 → less vram; 3.0 → high quality"
                }),
                "local_range": ("INT", {
                    "default": 11,
                    "min": 9,
                    "max": 11,
                    "step": 2,
                    "tooltip": "Recommended: 9 or 11\nlocal_range=9 → sharper details; 11 → more stable results"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624
                }),
                "device": (device_choices, {
                    "default": device_choices[0],
                    "tooltip": "Device to load the weights, default: auto (CUDA if available, else CPU)"
                }),
                "precision": (["fp16", "bf16"], {
                    "default": "bf16",
                    "tooltip": "Data and inference precision."
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention"], {
                    "default": "sparse_sage_attention",
                    "tooltip": '"sparse_sage_attention" is available for sm_75 to sm_120\n"block_sparse_attention" is available for sm_80 to sm_100'
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'Download the entire "FlashVSR" folder with all the files inside it from "https://huggingface.co/JunhaoZhuang/FlashVSR" and put it in the "ComfyUI/models"'
    
    def main(self, frames, mode, scale, target_height, target_width, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, sparse_ratio, kv_ratio, local_range, seed, device, precision, attention_mode):
        _device = device
        if device == "auto":
            _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else device
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
        
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
        
        if tiled_dit and (tile_overlap > tile_size / 2):
            raise ValueError('The "tile_overlap" must be less than half of "tile_size"!')
        
        if attention_mode == "sparse_sage_attention":
            wan_video_dit.USE_BLOCK_ATTN = False
        else:
            wan_video_dit.USE_BLOCK_ATTN = True
        
        _frames = frames
        if target_width > 0 and target_height > 0:
            if target_width % scale != 0 or target_height % scale != 0:
                log(f"Warning: target_width or target_height is not divisible by scale. This may lead to slight dimension mismatch.", message_type='warning')
            
            # NHWC to NCHW
            _frames_nchw = _frames.permute(0, 3, 1, 2)
            
            pre_H = target_height // scale
            pre_W = target_width // scale
            
            _frames_nchw = F.interpolate(_frames_nchw, size=(pre_H, pre_W), mode='bicubic', align_corners=False)
            
            # NCHW back to NHWC
            _frames = _frames_nchw.permute(0, 2, 3, 1)

        if frames.shape[0] < 21:
            add = 21 - frames.shape[0]
            last_frame = frames[-1:, :, :, :]
            padding_frames = last_frame.repeat(add, 1, 1, 1)
            _frames = torch.cat([frames, padding_frames], dim=0)
            #raise ValueError(f"Number of frames must be at least 21, got {frames.shape[0]}")
        
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        try:
            dtype = dtype_map[precision]
        except:
            dtype = torch.bfloat16

        if tiled_dit:
            N, H, W, C = _frames.shape
            num_aligned_frames = largest_8n1_leq(N + 4) - 4
            
            # Use uint8 for output canvas to save RAM
            final_output_canvas = torch.zeros(
                (num_aligned_frames, H * scale, W * scale, C), 
                dtype=torch.float32,  # Use float32 for accumulation
                device="cpu"
            )
            weight_sum_canvas = torch.zeros_like(final_output_canvas)
            tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
            latent_tiles_cpu = []
            
            pipe = init_pipeline(mode, _device, dtype)
            
            for i, (x1, y1, x2, y2) in enumerate(cqdm(tile_coords, desc="Processing Tiles")):
                log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: coords ({x1},{y1}) to ({x2},{y2})", message_type='info')
                input_tile = _frames[:, y1:y2, x1:x2, :]
                
                LQ_tile, th, tw, num_frames_for_pipe = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
                if "long" not in mode:
                    LQ_tile = LQ_tile.to(_device)
                
                output_tile_gpu = pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                    LQ_video=LQ_tile, num_frames=num_frames_for_pipe, height=th, width=tw, is_full_block=False, if_buffer=True,
                    topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                    color_fix=color_fix, unload_dit=unload_dit
                )
                
                processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
                
                mask_nchw = create_feather_mask(
                    (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                    tile_overlap * scale
                ).to("cpu")
                mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
                out_x1, out_y1 = x1 * scale, y1 * scale
                
                tile_H_scaled = processed_tile_cpu.shape[1]
                tile_W_scaled = processed_tile_cpu.shape[2]
                out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
                final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
                weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
                
                del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
                clean_vram()
                torch.cuda.empty_cache()
                gc.collect()
                
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            # final_output = final_output_canvas / weight_sum_canvas # This creates a third large tensor, causing OOM.
            # Use in-place division to avoid the memory spike.
            torch.divide(final_output_canvas, weight_sum_canvas, out=final_output_canvas)
            
            # Keep full precision instead of quantizing to uint8
            final_output = final_output_canvas.clamp(0, 1)
            
            del weight_sum_canvas
            gc.collect()
        else:
            log("[FlashVSR] Preparing frames...")
            LQ, th, tw, num_frames_for_pipe = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype)
            if "long" not in mode:
                LQ = LQ.to(_device)
            
            pipe = init_pipeline(mode, _device, dtype)
            log(f"[FlashVSR] Processing {frames.shape[0]} frames...", message_type='info')
            
            video = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                progress_bar_cmd=cqdm, LQ_video=LQ, num_frames=num_frames_for_pipe, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix = color_fix, unload_dit=unload_dit
            )
            
            # Keep full precision
            final_output = tensor2video(video).to('cpu').clamp(0, 1)
            
            del pipe, video, LQ
            clean_vram()
        
        log("[FlashVSR] Done.", message_type='info')
        
        # Resize to exact target dimensions if specified
        if target_width > 0 and target_height > 0:
            current_output = final_output[:frames.shape[0], :, :, :]
            # NHWC to NCHW for interpolate
            current_output_nchw = current_output.permute(0, 3, 1, 2)
            resized_output_nchw = F.interpolate(
                current_output_nchw, 
                size=(target_height, target_width), 
                mode='bicubic', 
                align_corners=False
            )
            # NCHW back to NHWC
            final_output = resized_output_nchw.permute(0, 2, 3, 1).clamp(0, 1)
            del current_output, current_output_nchw, resized_output_nchw
            clean_vram()
            log(f"[FlashVSR] Resized output to exact target: {target_width}x{target_height}", message_type='info')
        else:
            final_output = final_output[:frames.shape[0], :, :, :]
        
        if frames.shape[0] == 1:
            final_output = final_output.to(_device)
            stacked_image_tensor = torch.median(final_output, dim=0).unsqueeze(0).to('cpu')
            del final_output
            clean_vram()
            return (stacked_image_tensor,)
        
        return (final_output,)

class FlashVSRStreamingNode:
    """FlashVSR node that streams output directly to video file, saving RAM"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE", {
                    "tooltip": "Sequential video frames as IMAGE tensor batch"
                }),
                "mode": (["tiny", "tiny-long", "full"], {
                    "default": "tiny",
                    "tooltip": 'Using "tiny-long" mode can significantly reduce VRAM used with long video input.'
                }),
                "scale": ("INT", {
                    "default": 2,
                    "min": 2,
                    "max": 4,
                }),
                "target_height": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 4,
                    "tooltip": "Target output height. 0 for auto (input height * scale)."
                }),
                "target_width": ("INT", {
                    "default": 0, "min": 0, "max": 8192, "step": 4,
                    "tooltip": "Target output width. 0 for auto (input width * scale)."
                }),
                "frame_rate": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 120,
                    "tooltip": "Output video frame rate"
                }),
                "filename_prefix": ("STRING", {
                    "default": "FlashVSR",
                    "tooltip": "Prefix for output video filename"
                }),
                "video_format": (["mp4", "webm", "mkv", "avi"], {
                    "default": "mp4",
                    "tooltip": "Output video format"
                }),
                "quality": ("INT", {
                    "default": 23,
                    "min": 0,
                    "max": 51,
                    "tooltip": "Video quality (CRF). Lower = better quality. 23 is recommended."
                }),
                "pix_fmt": (["yuv420p", "yuv444p", "rgb24"], {
                    "default": "yuv444p",
                    "tooltip": "Pixel format. yuv420p: standard (smaller file, color subsampling). yuv444p: better color quality. rgb24: lossless color (larger file, not all players support)."
                }),
                "color_fix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use wavelet transform to correct output video color."
                }),
                "tiled_vae": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable tiling: faster decode but higher VRAM usage.\nSet to True for lower memory consumption at the cost of speed."
                }),
                "unload_dit": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Unload DiT before decoding to reduce VRAM peak at the cost of speed."
                }),
                "batch_size": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Number of frames to process at once. Lower values use less RAM but take longer."
                }),
                "sparse_ratio": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.5,
                    "max": 2.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.5 or 2.0\n1.5 → faster; 2.0 → more stable"
                }),
                "kv_ratio": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Recommended: 1.0 to 3.0\n1.0 → less vram; 3.0 → high quality"
                }),
                "local_range": ("INT", {
                    "default": 11,
                    "min": 9,
                    "max": 11,
                    "step": 2,
                    "tooltip": "Recommended: 9 or 11\nlocal_range=9 → sharper details; 11 → more stable results"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1125899906842624
                }),
                "device": (device_choices, {
                    "default": device_choices[0],
                    "tooltip": "Device to load the weights, default: auto (CUDA if available, else CPU)"
                }),
                "precision": (["fp16", "bf16"], {
                    "default": "bf16",
                    "tooltip": "Data and inference precision."
                }),
                "attention_mode": (["sparse_sage_attention", "block_sparse_attention"], {
                    "default": "sparse_sage_attention",
                    "tooltip": '"sparse_sage_attention" is available for sm_75 to sm_120\n"block_sparse_attention" is available for sm_80 to sm_100'
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Optional audio track to add to the output video"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    OUTPUT_NODE = True
    FUNCTION = "process_streaming"
    CATEGORY = "FlashVSR"
    DESCRIPTION = 'FlashVSR with streaming output - processes video in batches and streams directly to file, saving RAM for long videos'
    
    def process_streaming(self, frames, mode, scale, target_height, target_width, frame_rate, filename_prefix, 
                         video_format, quality, pix_fmt, color_fix, tiled_vae, unload_dit, batch_size,
                         sparse_ratio, kv_ratio, local_range, seed, device, precision, attention_mode, audio=None):
        
        ffmpeg_path = get_ffmpeg_path()
        if ffmpeg_path is None:
            raise ProcessLookupError("ffmpeg is required for streaming output but could not be found.")
        
        _device = device
        if device == "auto":
            _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else device
        if _device == "auto" or _device not in device_choices:
            raise RuntimeError("No devices found to run FlashVSR!")
        
        if _device.startswith("cuda"):
            torch.cuda.set_device(_device)
        
        if attention_mode == "sparse_sage_attention":
            wan_video_dit.USE_BLOCK_ATTN = False
        else:
            wan_video_dit.USE_BLOCK_ATTN = True
        
        _frames = frames
        if target_width > 0 and target_height > 0:
            if target_width % scale != 0 or target_height % scale != 0:
                log(f"Warning: target_width or target_height is not divisible by scale. This may lead to slight dimension mismatch.", message_type='warning')
            
            _frames_nchw = _frames.permute(0, 3, 1, 2)
            pre_H = target_height // scale
            pre_W = target_width // scale
            _frames_nchw = F.interpolate(_frames_nchw, size=(pre_H, pre_W), mode='bicubic', align_corners=False)
            _frames = _frames_nchw.permute(0, 2, 3, 1)

        if frames.shape[0] < 21:
            add = 21 - frames.shape[0]
            last_frame = frames[-1:, :, :, :]
            padding_frames = last_frame.repeat(add, 1, 1, 1)
            _frames = torch.cat([frames, padding_frames], dim=0)
        
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        dtype = dtype_map.get(precision, torch.bfloat16)
        
        # Setup output path
        output_dir = folder_paths.get_output_directory()
        counter = 1
        full_output_folder = output_dir
        filename = filename_prefix
        
        # Find next available counter
        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter >= counter:
                    counter = file_counter + 1
        
        file = f"{filename}_{counter:05}.{video_format}"
        file_path = os.path.join(full_output_folder, file)
        
        N, H, W, C = _frames.shape
        
        # Get actual output resolution from prepare_input_tensor
        # This accounts for FlashVSR's internal alignment (128 multiple)
        _, actual_output_H, actual_output_W, _ = prepare_input_tensor(_frames[:1], _device, scale=scale, dtype=dtype)
        
        # Determine final output resolution
        # If target dimensions are specified, use them; otherwise use aligned dimensions
        if target_width > 0 and target_height > 0:
            final_output_W = target_width
            final_output_H = target_height
            need_resize = (final_output_W != actual_output_W or final_output_H != actual_output_H)
            log(f"[FlashVSR Streaming] Input resolution: {W}x{H}", message_type='info')
            log(f"[FlashVSR Streaming] FlashVSR output resolution (aligned): {actual_output_W}x{actual_output_H}", message_type='info')
            log(f"[FlashVSR Streaming] Final output resolution (target): {final_output_W}x{final_output_H}", message_type='info')
        else:
            final_output_W = actual_output_W
            final_output_H = actual_output_H
            need_resize = False
            log(f"[FlashVSR Streaming] Input resolution: {W}x{H}", message_type='info')
            log(f"[FlashVSR Streaming] Expected output resolution (after alignment): {actual_output_W}x{actual_output_H}", message_type='info')
        
        # Setup FFmpeg process for streaming
        codec_map = {
            "mp4": "libx264",
            "webm": "libvpx-vp9",
            "mkv": "libx264",
            "avi": "libx264"
        }
        codec = codec_map.get(video_format, "libx264")
        
        ffmpeg_cmd = [
            ffmpeg_path, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{final_output_W}x{final_output_H}", "-r", str(frame_rate), "-i", "-",
            "-c:v", codec, "-crf", str(quality),
        ]
        
        # Add pixel format and color settings
        if pix_fmt == "rgb24":
            # RGB24: lossless color, larger file
            ffmpeg_cmd.extend(["-pix_fmt", "rgb24"])
        elif pix_fmt == "yuv444p":
            # YUV444: no chroma subsampling, better color quality
            ffmpeg_cmd.extend(["-pix_fmt", "yuv444p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "iec61966-2-1"])
        else:
            # YUV420: standard, smaller file with chroma subsampling
            ffmpeg_cmd.extend(["-pix_fmt", "yuv420p"])
        
        ffmpeg_cmd.append(file_path)
        
        log(f"[FlashVSR Streaming] Starting video output: {file}", message_type='info')
        log(f"[FlashVSR Streaming] FFmpeg configured for: {final_output_W}x{final_output_H}, FPS: {frame_rate}", message_type='info')
        
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
        
        try:
            pipe = init_pipeline(mode, _device, dtype)
            
            # Process in batches to save memory
            log("[FlashVSR Streaming] Processing video in batches to save memory...")
            
            total_frames = frames.shape[0]
            frames_written = 0
            
            # Calculate overlap to ensure continuity (FlashVSR needs context)
            # We need at least 4 frames overlap for FlashVSR's temporal consistency
            overlap_frames = 8
            
            # Process in batches
            batch_start = 0
            while batch_start < total_frames:
                batch_end = min(batch_start + batch_size, total_frames)
                
                # Extend batch to include overlap for next batch (except last batch)
                batch_end_with_overlap = batch_end
                if batch_end < total_frames:
                    batch_end_with_overlap = min(batch_end + overlap_frames, total_frames)
                
                log(f"[FlashVSR Streaming] Processing frames {batch_start} to {batch_end_with_overlap} (will output {batch_start} to {batch_end})...", message_type='info')
                
                # Extract batch
                batch_frames = _frames[batch_start:batch_end_with_overlap]
                original_batch_size = batch_frames.shape[0]
                
                # FlashVSR uses 8n+1 alignment which may reduce output frames
                # Calculate expected output: largest_8n1_leq(N+4) - 4 (approximately)
                frames_needed = batch_end - batch_start
                
                # To ensure we get enough output frames, we may need to pad input
                # We need: largest_8n1_leq(input_frames+4) - 4 >= frames_needed
                # So: input_frames+4 >= frames_needed+4+padding_for_8n1
                # Adding 8 extra frames should be safe
                min_input_for_output = frames_needed + 8
                
                # Ensure minimum 21 frames for FlashVSR
                min_frames = max(21, min_input_for_output)
                
                if batch_frames.shape[0] < min_frames:
                    add = min_frames - batch_frames.shape[0]
                    last_frame = batch_frames[-1:, :, :, :]
                    padding_frames = last_frame.repeat(add, 1, 1, 1)
                    batch_frames = torch.cat([batch_frames, padding_frames], dim=0)
                    log(f"[FlashVSR Streaming] Padded batch from {original_batch_size} to {batch_frames.shape[0]} frames (need {frames_needed} output frames)", message_type='info')
                
                # Prepare input
                LQ, th, tw, num_frames_for_pipe = prepare_input_tensor(batch_frames, _device, scale=scale, dtype=dtype)
                if "long" not in mode:
                    LQ = LQ.to(_device)
                
                # Process batch
                video = pipe(
                    prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                    LQ_video=LQ, num_frames=num_frames_for_pipe, height=th, width=tw, 
                    is_full_block=False, if_buffer=True,
                    topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                    color_fix=color_fix, unload_dit=unload_dit
                )
                
                video_tensor = tensor2video(video).to('cpu')
                
                # Calculate how many frames to write from this batch
                # For the last batch, we might have fewer frames than batch_size
                frames_to_write = batch_end - batch_start
                
                # The video_tensor should contain the processed frames
                # We need to write exactly frames_to_write frames (the frames we originally wanted)
                # But we can't write more frames than video_tensor has
                actual_frames_in_batch = min(frames_to_write, video_tensor.shape[0])
                
                # If video_tensor doesn't have enough frames, warn about it
                if video_tensor.shape[0] < frames_to_write:
                    log(f"[FlashVSR Streaming] Warning: video_tensor only has {video_tensor.shape[0]} frames but need {frames_to_write}", message_type='warning')
                
                # Log batch info
                log(f"[FlashVSR Streaming] Batch {batch_start}-{batch_end}: video_tensor shape {video_tensor.shape}, num_frames_for_pipe {num_frames_for_pipe}, will write {actual_frames_in_batch} frames", message_type='info')
                
                # Debug first batch
                if batch_start == 0:
                    log(f"[FlashVSR Debug] Batch output range: [{video_tensor.min():.3f}, {video_tensor.max():.3f}]", message_type='info')
                
                # Write frames from this batch
                for frame_idx in range(actual_frames_in_batch):
                    frame = video_tensor[frame_idx].clamp(0, 1)
                    
                    # Resize frame if needed
                    if need_resize:
                        # HWC to CHW to BCHW
                        frame_nchw = frame.permute(2, 0, 1).unsqueeze(0)
                        frame_resized_nchw = F.interpolate(
                            frame_nchw,
                            size=(final_output_H, final_output_W),
                            mode='bicubic',
                            align_corners=False
                        )
                        # BCHW to CHW to HWC
                        frame = frame_resized_nchw.squeeze(0).permute(1, 2, 0).clamp(0, 1)
                    
                    # Debug first frame
                    if frames_written == 0:
                        log(f"[FlashVSR Debug] Frame shape: {frame.shape}, range: [{frame.min():.3f}, {frame.max():.3f}]", message_type='info')
                    
                    frame_bytes = tensor_to_bytes(frame)
                    proc.stdin.write(frame_bytes.tobytes())
                    
                    frames_written += 1
                    if frames_written % 50 == 0:
                        log(f"[FlashVSR Streaming] Written {frames_written}/{total_frames} frames", message_type='info')
                        # Flush to ensure data is sent to FFmpeg
                        try:
                            proc.stdin.flush()
                        except:
                            pass
                
                # Clean up batch
                del video, video_tensor, LQ, batch_frames
                clean_vram()
                gc.collect()
                
                # Move to next batch (without overlap)
                batch_start = batch_end
            
            log(f"[FlashVSR Streaming] Total frames written: {frames_written}/{total_frames}", message_type='finish')
            
            if frames_written < total_frames:
                log(f"[FlashVSR Streaming] Warning: Only {frames_written} frames written, expected {total_frames}", message_type='warning')
            
            # Close FFmpeg input and wait for encoding to complete
            log("[FlashVSR Streaming] Closing FFmpeg input and waiting for encoding to complete...", message_type='info')
            
            # Use communicate() instead of wait() to avoid deadlock
            # communicate() will automatically close stdin and read stderr
            # Do NOT manually close stdin before calling communicate()
            try:
                _, stderr_output = proc.communicate(timeout=300)  # 5 minute timeout
                stderr_output = stderr_output.decode('utf-8', errors='ignore')
            except subprocess.TimeoutExpired:
                log("[FlashVSR Streaming] FFmpeg timeout, killing process...", message_type='error')
                proc.kill()
                _, stderr_output = proc.communicate()
                stderr_output = stderr_output.decode('utf-8', errors='ignore')
                raise Exception(f"FFmpeg timed out after 5 minutes. Output:\n{stderr_output}")
            
            if proc.returncode != 0:
                log(f"[FlashVSR Streaming] FFmpeg failed with code {proc.returncode}", message_type='error')
                log(f"FFmpeg output:\n{stderr_output}", message_type='error')
                raise Exception(f"FFmpeg error (code {proc.returncode}): {stderr_output}")
            
            # Print ffmpeg info (not error)
            if stderr_output:
                log(f"[FlashVSR Streaming] FFmpeg completed successfully", message_type='info')
                # Only show last part to avoid flooding logs
                if len(stderr_output) > 1000:
                    log(f"FFmpeg output (last 1000 chars): {stderr_output[-1000:]}", message_type='info')
                else:
                    log(f"FFmpeg output: {stderr_output}", message_type='info')
            
            log(f"[FlashVSR Streaming] Video saved: {file_path}", message_type='finish')
            
            # Add audio if provided
            final_output_path = file_path
            if audio is not None:
                try:
                    audio_waveform = audio.get('waveform')
                    if audio_waveform is not None and audio_waveform.numel() > 0:
                        log(f"[FlashVSR Streaming] Adding audio to video...", message_type='info')
                        
                        # Create output file with audio
                        file_base, file_ext = os.path.splitext(file)
                        file_with_audio = f"{file_base}-audio{file_ext}"
                        output_file_with_audio_path = os.path.join(full_output_folder, file_with_audio)
                        
                        # Get audio info
                        sample_rate = audio.get('sample_rate', 44100)
                        channels = audio_waveform.size(1)
                        
                        # Calculate minimum audio duration using frames_written from batch processing
                        total_frames_output = frames_written
                        min_audio_dur = total_frames_output / frame_rate + 1
                        
                        # Prepare audio padding
                        apad = ["-af", f"apad=whole_dur={min_audio_dur}"]
                        
                        # FFmpeg command to mux audio with video
                        mux_args = [
                            ffmpeg_path, "-v", "error", "-y", 
                            "-i", file_path,  # Input video
                            "-ar", str(sample_rate), 
                            "-ac", str(channels),
                            "-f", "f32le", 
                            "-i", "-",  # Input audio from stdin
                            "-c:v", "copy",  # Copy video stream
                            "-c:a", "aac",   # Encode audio to AAC
                            "-b:a", "192k"   # Audio bitrate
                        ] + apad + ["-shortest", output_file_with_audio_path]
                        
                        # Prepare audio data
                        audio_data = audio_waveform.squeeze(0).transpose(0, 1).numpy().tobytes()
                        
                        # Run ffmpeg to add audio
                        result = subprocess.run(
                            mux_args, 
                            input=audio_data,
                            capture_output=True, 
                            check=True
                        )
                        
                        if result.stderr:
                            log(f"[FlashVSR Streaming] Audio mux info: {result.stderr.decode('utf-8', errors='ignore')[-300:]}", message_type='info')
                        
                        final_output_path = output_file_with_audio_path
                        log(f"[FlashVSR Streaming] Video with audio saved: {output_file_with_audio_path}", message_type='finish')
                    else:
                        log(f"[FlashVSR Streaming] Audio input is empty, skipping audio", message_type='warning')
                except Exception as e:
                    log(f"[FlashVSR Streaming] Failed to add audio: {str(e)}", message_type='warning')
                    # Continue with video-only output
            
        except Exception as e:
            proc.kill()
            raise e
        finally:
            if proc.poll() is None:
                proc.kill()
        
        return (final_output_path,)


NODE_CLASS_MAPPINGS = {
    "FlashVSRNode": FlashVSRNode,
    "FlashVSRStreamingNode": FlashVSRStreamingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashVSRNode": "FlashVSR Ultra-Fast",
    "FlashVSRStreamingNode": "FlashVSR Streaming (Low RAM)",
}
