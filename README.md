# ComfyUI-FlashVSR_Ultra_Long
Ultra-fast FlashVSR with streaming support for long videos (2 min+). Process videos of any length with minimal RAM usage.   

## Preview
![](./img/demo.gif)

## Usage
- **mode:**  
`tiny` -> faster (default); `full` -> higher quality  
- **scale:**  
`4` is always better, unless you are low on VRAM then use `2`    
- **color_fix:**  
Use wavelet transform to correct the color of output video.  
- **tiled_vae:**  
Set to True for lower VRAM consumption during decoding at the cost of speed.  
- **tiled_dit:**  
Significantly reduces VRAM usage at the cost of speed.
- **tile\_size, tile\_overlap**:  
How to split the input video.  
- **unload_dit:**
Unload DiT before decoding to reduce VRAM peak at the cost of speed.

**Note:** Parameters `tiled_dit`, `tile_size`, and `tile_overlap` are only available in the **FlashVSR Ultra-Fast** node. The **FlashVSR Streaming** node uses batch processing instead of spatial tiling for memory efficiency with long videos.

## Features

### FlashVSR Ultra-Fast
Standard node for processing videos with tiling support to reduce VRAM usage.

### FlashVSR Streaming (Low RAM)
Streaming node that processes video in batches and streams output directly to file, enabling processing of videos of any length with minimal RAM usage. Perfect for long videos that would otherwise cause out-of-memory errors.

**Note:** The streaming node uses batch processing instead of tiling (`tiled_dit` is not available), making it optimized for long videos rather than spatial tiling.

**Streaming Node Benefits:**
- Process videos of unlimited length
- Minimal RAM usage (processes in configurable batches)
- Direct-to-file output (no intermediate storage)
- Configurable video formats (MP4, WebM, MKV, AVI)
- Audio track support
- Real-time progress tracking

**Streaming Node Parameters:**
- **batch_size:** Number of frames to process at once (default: 64, range: 16-512)
- **frame_rate:** Output video frame rate (default: 30, range: 1-120)
- **filename_prefix:** Prefix for output video filename (default: "FlashVSR")
- **video_format:** Output video format (MP4, WebM, MKV, AVI)
- **quality:** Video quality (CRF, default: 23, lower = better quality)
- **pix_fmt:** Pixel format (yuv420p, yuv444p, rgb24)

## Installation

#### nodes: 

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Long.git
python -m pip install -r ComfyUI-FlashVSR_Ultra_Long/requirements.txt
```

#### models:

- Download the entire `FlashVSR` folder with all the files inside it from [here](https://huggingface.co/JunhaoZhuang/FlashVSR) and put it in the `ComfyUI/models`

```
├── ComfyUI/models/FlashVSR
|     ├── LQ_proj_in.ckpt
|     ├── TCDecoder.ckpt
|     ├── diffusion_pytorch_model_streaming_dmd.safetensors
|     ├── Wan2.1_VAE.pth
```

## Acknowledgments
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) @OpenImagingLab  
- [ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) @lihaoyun6
- [Sparse_SageAttention](https://github.com/jt-zhang/Sparse_SageAttention_API) @jt-zhang
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) @comfyanonymous
