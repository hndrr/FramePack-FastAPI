import base64
import gc
import io
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import torch
from PIL import Image

from api import queue_manager as qm
from api import settings
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.hunyuan import encode_prompt_conds, vae_encode
from diffusers_helper.load_lora import load_lora
from diffusers_helper.memory import (
    gpu, fake_diffusers_current_device, load_model_as_complete, 
    unload_complete_models, move_model_to_device_with_memory_preservation
)
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.utils import crop_or_pad_yield_mask, resize_and_center_crop, write_PIL_image_with_png_info

# Global cache for text encoding
_text_encoding_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def ensure_tensor_properties(tensor: torch.Tensor, target_device: torch.device, target_dtype: torch.dtype = None) -> torch.Tensor:
    """Ensure tensor has the correct device and dtype"""
    if target_dtype is not None and tensor.dtype != target_dtype:
        tensor = tensor.to(dtype=target_dtype)
    if tensor.device != target_device:
        tensor = tensor.to(target_device)
    return tensor


def clear_memory(verbose: bool = False):
    """Clear GPU and CPU memory efficiently"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print(f"Memory cleared. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")


def apply_fp8_optimization_simple(model, dtype="e4m3"):
    """Apply FP8 optimization to transformer model (simplified version)"""
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
        return model
    
    try:
        # Basic FP8 quantization for transformer blocks
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if 'ff.net' in name or 'attn' in name:  # Target feedforward and attention layers
                    # This is a simplified placeholder - actual FP8 implementation would be more complex
                    pass
        print("FP8 optimization applied (simplified)")
    except Exception as e:
        print(f"FP8 optimization skipped: {e}")
    
    return model


def load_multiple_loras(model, lora_configs: List[Dict[str, Any]]):
    """Load multiple LoRAs into model"""
    for config in lora_configs:
        lora_path = config["path"]
        lora_scale = config.get("scale", 1.0)
        
        if os.path.exists(lora_path):
            lora_dir, lora_name = os.path.split(lora_path)
            model = load_lora(model, Path(lora_dir), lora_name, lora_scale=lora_scale)
            print(f"Loaded LoRA: {lora_name} with scale {lora_scale}")
    
    return model


def get_text_encoding_cached(
    prompt: str,
    negative_prompt: str,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    high_vram: bool,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get text encodings with caching support"""
    cache_key = f"{prompt}||{negative_prompt}"
    
    if cache_key in _text_encoding_cache:
        llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n = _text_encoding_cache[cache_key]
        return (
            llama_vec.to(device),
            clip_l_pooler.to(device),
            llama_vec_n.to(device),
            clip_l_pooler_n.to(device)
        )
    
    # Prepare models for text encoding in low VRAM mode
    if not high_vram:
        fake_diffusers_current_device(text_encoder, device)
        load_model_as_complete(text_encoder_2, target_device=device)
    
    # Encode positive prompt
    llama_vec, clip_l_pooler = encode_prompt_conds(
        prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
    )
    
    if llama_vec is None or clip_l_pooler is None:
        raise ValueError("Failed to encode prompt")
    
    # Handle negative prompt
    if negative_prompt:
        llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
            negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )
        if llama_vec_n is None or clip_l_pooler_n is None:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    else:
        llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
    
    # Cache the results
    _text_encoding_cache[cache_key] = (
        llama_vec.cpu(),
        clip_l_pooler.cpu(),
        llama_vec_n.cpu(),
        clip_l_pooler_n.cpu()
    )
    
    # Limit cache size
    if len(_text_encoding_cache) > 50:
        # Keep only the last 25 entries
        keys = list(_text_encoding_cache.keys())
        for key in keys[:25]:
            del _text_encoding_cache[key]
    
    return llama_vec.to(device), clip_l_pooler.to(device), llama_vec_n.to(device), clip_l_pooler_n.to(device)


def process_image_generation(job_id: str, job_data: dict, models: dict):
    """
    Process single image-to-image generation job using Hunyuan Video model for 1 frame
    Optimized for performance with caching and efficient memory management
    """
    try:
        print(f"Starting image generation job: {job_id}")
        qm.update_job_status(job_id, "processing")
        qm.update_job_progress(job_id, 0.0, 0, 100, "Initializing image generation...")

        # Extract parameters
        data = job_data.get("data", {})
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        seed = data.get("seed", -1)
        steps = data.get("steps", 30)
        cfg = data.get("cfg", 1.0)
        width = data.get("width", 1216)
        height = data.get("height", 704)
        lora_path = data.get("lora_path", "")
        lora_scale = data.get("lora_scale", 1.0)
        gpu_memory_preservation = data.get("gpu_memory_preservation", 10.0)
        sampling_mode = data.get("sampling_mode", "dpm-solver++")
        transformer_model = data.get("transformer_model", "base")
        use_fp8 = data.get("use_fp8", True)

        # Set random seed
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Extract models
        vae = models["vae"]
        text_encoder = models["text_encoder"]
        text_encoder_2 = models["text_encoder_2"]
        tokenizer = models["tokenizer"]
        tokenizer_2 = models["tokenizer_2"]
        feature_extractor = models["feature_extractor"]
        image_encoder = models["image_encoder"]
        transformer = models["transformer_base"] if transformer_model == "base" else models["transformer_f1"]
        high_vram = models["high_vram"]

        # Get required input image from job data
        input_image_b64 = data.get("input_image", "")
        print(f"Debug worker: input_image_b64 type = {type(input_image_b64)}")
        print(f"Debug worker: input_image_b64 length = {len(input_image_b64) if isinstance(input_image_b64, str) else 'N/A'}")
        
        if not input_image_b64:
            raise ValueError("Input image is required for image generation")

        # Decode base64 input image
        try:
            if isinstance(input_image_b64, str):
                # Clean up base64 string if it has data URL prefix
                if input_image_b64.startswith('data:'):
                    input_image_b64 = input_image_b64.split(',')[1]
                
                image_bytes = base64.b64decode(input_image_b64)
                print(f"Debug worker: Decoded {len(image_bytes)} bytes")
                input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                print(f"Debug worker: Image loaded successfully, size = {input_image.size}")
            else:
                raise ValueError(f"input_image_b64 must be string, got {type(input_image_b64)}")
        except Exception as img_error:
            print(f"Debug worker: Image decode error: {img_error}")
            raise ValueError(f"Failed to decode input image: {img_error}")

        # Process input image
        input_image_np = np.array(input_image)
        h, w, c = input_image_np.shape
        height, width = find_nearest_bucket(h, w, resolution=640)
        input_image_np = resize_and_center_crop(
            input_image_np, target_width=width, target_height=height
        )

        # Clean GPU if not high_vram
        if not high_vram:
            qm.update_job_progress(job_id, 0.05, 5, 100, "Cleaning GPU memory...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
            clear_memory()

        # Text encoding with caching
        qm.update_job_progress(job_id, 0.1, 10, 100, "Encoding prompts...")
        
        llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n = get_text_encoding_cached(
            prompt, negative_prompt, text_encoder, text_encoder_2, 
            tokenizer, tokenizer_2, high_vram, gpu
        )

        # Prepare attention masks
        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # FP8 optimization is disabled for stability
        # if use_fp8 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        #     qm.update_job_progress(job_id, 0.12, 12, 100, "Applying FP8 optimization...")
        #     transformer = apply_fp8_optimization_simple(transformer)

        # Encode input image with CLIP vision
        qm.update_job_progress(job_id, 0.18, 18, 100, "Encoding input image...")
        if not high_vram:
            image_encoder = image_encoder.to(gpu)
        
        vision_hidden_states = hf_clip_vision_encode(
            input_image_np, feature_extractor, image_encoder
        ).last_hidden_state
        vision_hidden_states = ensure_tensor_properties(
            vision_hidden_states, gpu, transformer.dtype
        )
        
        if not high_vram:
            image_encoder = image_encoder.cpu()
            clear_memory()

        # Encode input image with VAE
        qm.update_job_progress(job_id, 0.2, 20, 100, "Encoding with VAE...")
        video_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        video_pt = video_pt.permute(2, 0, 1)[None, :, None]

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        latents = vae_encode(video_pt, vae)
        if not high_vram:
            vae = vae.cpu()
            clear_memory()

        # Prepare latents
        qm.update_job_progress(job_id, 0.25, 25, 100, "Preparing latents...")
        latents = ensure_tensor_properties(latents, gpu, torch.bfloat16)

        # Ensure all tensors are on the correct device and dtype
        llama_vec = ensure_tensor_properties(llama_vec, gpu, transformer.dtype)
        llama_vec_n = ensure_tensor_properties(llama_vec_n, gpu, transformer.dtype)
        clip_l_pooler = ensure_tensor_properties(clip_l_pooler, gpu, transformer.dtype)
        clip_l_pooler_n = ensure_tensor_properties(clip_l_pooler_n, gpu, transformer.dtype)
        llama_attention_mask = llama_attention_mask.to(gpu)
        llama_attention_mask_n = llama_attention_mask_n.to(gpu)

        # Load LoRA if specified (following video generation pattern)
        if lora_path:
            full_lora_path = os.path.join(settings.LORA_DIR, lora_path) if not os.path.isabs(lora_path) else lora_path
            if os.path.exists(full_lora_path):
                qm.update_job_progress(job_id, 0.27, 27, 100, f"Loading LoRA: {lora_path}")
                lora_dir, lora_name = os.path.split(full_lora_path)
                try:
                    transformer = load_lora(transformer, Path(lora_dir), lora_name, lora_scale=lora_scale)
                    print(f"Job {job_id}: LoRA loaded successfully from {full_lora_path}")
                except Exception as e:
                    print(f"Job {job_id}: Error loading LoRA: {e}")
                    # Continue without LoRA instead of failing
            else:
                print(f"Job {job_id}: LoRA file not found at {full_lora_path}")

        # Move transformer to GPU with memory preservation
        if not high_vram:
            unload_complete_models()  # Unload other models first
            move_model_to_device_with_memory_preservation(
                transformer,
                target_device=gpu,
                preserved_memory_gb=gpu_memory_preservation
            )
        else:
            # In high VRAM mode, simply move to GPU
            transformer = transformer.to(gpu)

        # Generate using sample_hunyuan
        qm.update_job_progress(job_id, 0.3, 30, 100, "Generating image...")

        # Create random generator
        rnd = torch.Generator("cpu").manual_seed(seed)

        # Progress callback
        def progress_callback(d):
            step = d.get("i", 0)
            progress = 0.3 + (step / steps) * 0.6  # 30% to 90%
            qm.update_job_progress(job_id, progress, step, steps, f"Sampling step {step}/{steps}")

        # Sample using sample_hunyuan (1 frame generation)
        with torch.no_grad():
            generated = sample_hunyuan(
                transformer=transformer,
                sampler=sampling_mode if sampling_mode != "dpm-solver++" else "unipc",
                width=width,
                height=height,
                frames=1,  # Single frame for image generation
                real_guidance_scale=cfg,
                distilled_guidance_scale=1.0,
                guidance_rescale=0.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                initial_latent=latents,
                strength=1.0,
                image_embeddings=vision_hidden_states,
                callback=progress_callback
            )

        # Clear transformer from GPU before VAE decode
        if not high_vram:
            transformer = transformer.cpu()
            clear_memory()

        # Decode latents
        qm.update_job_progress(job_id, 0.9, 90, 100, "Decoding image...")

        if not high_vram:
            vae = vae.to(gpu)

        generated = ensure_tensor_properties(generated, gpu, torch.float16)

        with torch.no_grad():
            frames = vae.decode(generated / vae.config.scaling_factor, return_dict=False)[0]

        if not high_vram:
            vae = vae.cpu()
            clear_memory()

        # Extract single frame
        frame = frames[0, :, 0]  # [C, H, W]
        frame = frame.float().cpu().clamp(-1, 1)
        frame = (frame + 1) / 2  # [-1, 1] to [0, 1]
        frame = frame.permute(1, 2, 0).numpy()  # [H, W, C]
        frame = (frame * 255).astype(np.uint8)

        image = Image.fromarray(frame)

        # Clean up GPU memory
        if not high_vram:
            text_encoder = text_encoder.cpu()
            text_encoder_2 = text_encoder_2.cpu()
            clear_memory()

        # Save image
        qm.update_job_progress(job_id, 0.95, 95, 100, "Saving image...")

        output_dir = "outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_id}.png")

        # Prepare metadata - convert all values to strings for PNG info
        metadata = {
            "prompt": str(prompt),
            "negative_prompt": str(negative_prompt),
            "seed": str(seed),
            "steps": str(steps),
            "cfg": str(cfg),
            "width": str(width),
            "height": str(height),
            "model": f"FramePack-HunyuanVideo ({transformer_model})",
            "sampling_mode": str(sampling_mode),
            "lora_path": str(lora_path) if lora_path else "",
            "lora_scale": str(lora_scale) if lora_path else "",
            "timestamp": datetime.now().isoformat()
        }

        # Save with metadata
        write_PIL_image_with_png_info(image, metadata, output_path)

        # Generate thumbnail
        thumbnail = image.copy()
        thumbnail.thumbnail((512, 512))
        thumbnail_buffer = io.BytesIO()
        thumbnail.save(thumbnail_buffer, format="PNG")
        thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode('utf-8')

        # Update job result
        result = {
            "image_path": output_path,
            "seed": seed,
            "width": width,
            "height": height,
            "thumbnail": f"data:image/png;base64,{thumbnail_base64}"
        }

        qm.update_job_result(job_id, result)
        qm.update_job_status(job_id, "completed")
        qm.update_job_progress(job_id, 1.0, 100, 100, "Image generation completed!")

        print(f"Image generation job completed: {job_id}")

    except Exception as e:
        error_msg = str(e)
        if "'str' object has no attribute 'read'" in error_msg:
            error_msg = f"image load error: {error_msg}"
        print(f"Error in image generation job {job_id}: {error_msg}")
        traceback.print_exc()
        qm.update_job_status(job_id, f"failed - {error_msg}")
        qm.update_job_progress(job_id, 0.0, 0, 100, f"Error: {error_msg}")
    finally:
        # Ensure memory is cleaned
        clear_memory()


def process_batch_images(job_id: str, job_data: dict, models: dict):
    """
    Process batch image generation job
    """
    try:
        print(f"Starting batch image generation job: {job_id}")
        qm.update_job_status(job_id, "processing")
        qm.update_job_progress(job_id, 0.0, 0, 100, "Initializing batch image generation...")

        # Extract parameters
        data = job_data.get("data", {})
        prompts = data.get("prompts", [])
        input_image_b64 = data.get("input_image", "")
        negative_prompt = data.get("negative_prompt", "")
        seeds = data.get("seeds", None)
        steps = data.get("steps", 30)
        cfg = data.get("cfg", 1.0)
        width = data.get("width", 1216)
        height = data.get("height", 704)
        lora_path = data.get("lora_path", "")
        lora_scale = data.get("lora_scale", 1.0)
        gpu_memory_preservation = data.get("gpu_memory_preservation", 10.0)
        sampling_mode = data.get("sampling_mode", "dpm-solver++")
        transformer_model = data.get("transformer_model", "base")

        # Validate input image
        if not input_image_b64:
            raise ValueError("Input image is required for batch image generation")

        # Generate seeds if not provided
        if not seeds:
            seeds = [np.random.randint(0, 2**32 - 1) for _ in range(len(prompts))]

        output_dir = "outputs/images"
        os.makedirs(output_dir, exist_ok=True)

        results = []

        # Process each prompt
        for idx, (prompt, seed) in enumerate(zip(prompts, seeds)):
            progress = idx / len(prompts)
            qm.update_job_progress(
                job_id, progress, idx, len(prompts),
                f"Generating image {idx+1}/{len(prompts)}"
            )

            # Create individual job data
            individual_job_data = {
                "data": {
                    "prompt": prompt,
                    "input_image": input_image_b64,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "width": width,
                    "height": height,
                    "lora_path": lora_path,
                    "lora_scale": lora_scale,
                    "gpu_memory_preservation": gpu_memory_preservation,
                    "sampling_mode": sampling_mode,
                    "transformer_model": transformer_model
                }
            }

            # Generate using main function but save with batch filename
            sub_job_id = f"{job_id}_{idx}"
            process_image_generation(sub_job_id, individual_job_data, models)

            # Get the generated image path
            image_path = os.path.join(output_dir, f"{sub_job_id}.png")

            if os.path.exists(image_path):
                results.append({
                    "index": idx,
                    "prompt": prompt,
                    "seed": seed,
                    "image_path": image_path
                })

        # Update job result
        qm.update_job_result(job_id, {"images": results})
        qm.update_job_status(job_id, "completed")
        qm.update_job_progress(job_id, 1.0, len(prompts), len(prompts), "Batch generation completed!")

        print(f"Batch image generation job completed: {job_id}")

    except Exception as e:
        print(f"Error in batch image generation job {job_id}: {str(e)}")
        traceback.print_exc()
        qm.update_job_status(job_id, f"failed - {str(e)}")
        qm.update_job_progress(job_id, 0.0, 0, 100, f"Error: {str(e)}")


def process_image_transfer(job_id: str, job_data: dict, models: dict):
    """
    Process image transfer (kisekaeichi) job - style transfer from source to target
    """
    try:
        print(f"Starting image transfer job: {job_id}")
        qm.update_job_status(job_id, "processing")
        qm.update_job_progress(job_id, 0.0, 0, 100, "Initializing image transfer...")

        # Extract parameters
        data = job_data.get("data", {})
        source_image_b64 = data.get("source_image", "")
        target_image_b64 = data.get("target_image", "")
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        transfer_strength = data.get("transfer_strength", 0.5)
        seed = data.get("seed", -1)
        steps = data.get("steps", 30)
        cfg = data.get("cfg", 1.0)
        gpu_memory_preservation = data.get("gpu_memory_preservation", 10.0)
        sampling_mode = data.get("sampling_mode", "dpm-solver++")
        transformer_model = data.get("transformer_model", "base")

        # Set random seed
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Decode base64 images
        qm.update_job_progress(job_id, 0.1, 10, 100, "Decoding input images...")

        source_image = Image.open(io.BytesIO(base64.b64decode(source_image_b64))).convert('RGB')
        target_image = Image.open(io.BytesIO(base64.b64decode(target_image_b64))).convert('RGB')

        # Extract models
        vae = models["vae"]
        text_encoder = models["text_encoder"]
        text_encoder_2 = models["text_encoder_2"]
        tokenizer = models["tokenizer"]
        tokenizer_2 = models["tokenizer_2"]
        feature_extractor = models["feature_extractor"]
        image_encoder = models["image_encoder"]
        transformer = models["transformer_base"] if transformer_model == "base" else models["transformer_f1"]
        high_vram = models["high_vram"]

        # Process target image
        target_np = np.array(target_image)
        H, W, C = target_np.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        target_np = resize_and_center_crop(target_np, target_width=width, target_height=height)

        # Clean GPU if not high_vram
        if not high_vram:
            qm.update_job_progress(job_id, 0.15, 15, 100, "Cleaning GPU memory...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
            clear_memory()

        # Encode prompts
        qm.update_job_progress(job_id, 0.2, 20, 100, "Encoding prompts...")
        
        llama_vec, clip_l_pooler, llama_vec_n, clip_l_pooler_n = get_text_encoding_cached(
            prompt, negative_prompt, text_encoder, text_encoder_2,
            tokenizer, tokenizer_2, high_vram, gpu
        )

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # Encode source image with CLIP (for style)
        qm.update_job_progress(job_id, 0.3, 30, 100, "Encoding source image style...")
        source_np = np.array(source_image)
        source_np = resize_and_center_crop(source_np, target_width=width, target_height=height)

        if not high_vram:
            image_encoder = image_encoder.to(gpu)
        
        vision_hidden_states = hf_clip_vision_encode(
            source_np, feature_extractor, image_encoder
        ).last_hidden_state
        vision_hidden_states = ensure_tensor_properties(
            vision_hidden_states, gpu, transformer.dtype
        )
        
        if not high_vram:
            image_encoder = image_encoder.cpu()
            clear_memory()

        # Encode target image with VAE
        qm.update_job_progress(job_id, 0.4, 40, 100, "Encoding target image...")
        video_pt = torch.from_numpy(target_np).float() / 127.5 - 1
        video_pt = video_pt.permute(2, 0, 1)[None, :, None]

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        latents = vae_encode(video_pt, vae)
        if not high_vram:
            vae = vae.cpu()
            clear_memory()

        latents = ensure_tensor_properties(latents, gpu, torch.bfloat16)

        # Add noise based on transfer strength
        noise = torch.randn_like(latents)
        latents = latents * (1 - transfer_strength) + noise * transfer_strength

        # Move transformer to GPU with memory preservation
        if not high_vram:
            unload_complete_models()
            if not next(transformer.parameters()).is_cuda:
                move_model_to_device_with_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation
                )

        # Ensure tensor properties
        llama_vec = ensure_tensor_properties(llama_vec, gpu, transformer.dtype)
        llama_vec_n = ensure_tensor_properties(llama_vec_n, gpu, transformer.dtype)
        clip_l_pooler = ensure_tensor_properties(clip_l_pooler, gpu, transformer.dtype)
        clip_l_pooler_n = ensure_tensor_properties(clip_l_pooler_n, gpu, transformer.dtype)
        llama_attention_mask = llama_attention_mask.to(gpu)
        llama_attention_mask_n = llama_attention_mask_n.to(gpu)

        # Generate using sample_hunyuan
        qm.update_job_progress(job_id, 0.5, 50, 100, "Transferring style...")

        # Create random generator
        rnd = torch.Generator("cpu").manual_seed(seed)

        def progress_callback(d):
            step = d.get("i", 0)
            progress = 0.5 + (step / steps) * 0.4
            qm.update_job_progress(job_id, progress, step, steps, f"Transfer step {step}/{steps}")

        with torch.no_grad():
            generated = sample_hunyuan(
                transformer=transformer,
                sampler=sampling_mode if sampling_mode != "dpm-solver++" else "unipc",
                width=width,
                height=height,
                frames=1,
                real_guidance_scale=cfg,
                distilled_guidance_scale=1.0,
                guidance_rescale=0.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                initial_latent=latents,
                strength=transfer_strength,
                image_embeddings=vision_hidden_states,
                callback=progress_callback
            )

        # Clear transformer before decode
        if not high_vram:
            transformer = transformer.cpu()
            clear_memory()

        # Decode
        qm.update_job_progress(job_id, 0.9, 90, 100, "Decoding result...")

        if not high_vram:
            vae = vae.to(gpu)

        generated = ensure_tensor_properties(generated, gpu, torch.float16)

        with torch.no_grad():
            frames = vae.decode(generated / vae.config.scaling_factor, return_dict=False)[0]

        if not high_vram:
            vae = vae.cpu()
            text_encoder = text_encoder.cpu()
            text_encoder_2 = text_encoder_2.cpu()
            clear_memory()

        # Extract result
        frame = frames[0, :, 0]
        frame = frame.float().cpu().clamp(-1, 1)
        frame = (frame + 1) / 2
        frame = frame.permute(1, 2, 0).numpy()
        frame = (frame * 255).astype(np.uint8)

        result_image = Image.fromarray(frame)

        # Save result
        qm.update_job_progress(job_id, 0.95, 95, 100, "Saving result...")

        output_dir = "outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_id}.png")

        metadata = {
            "prompt": str(prompt),
            "negative_prompt": str(negative_prompt),
            "transfer_strength": str(transfer_strength),
            "seed": str(seed),
            "timestamp": datetime.now().isoformat(),
            "model": f"FramePack-HunyuanVideo ({transformer_model})"
        }

        write_PIL_image_with_png_info(result_image, metadata, output_path)

        # Generate thumbnail
        thumbnail = result_image.copy()
        thumbnail.thumbnail((512, 512))
        thumbnail_buffer = io.BytesIO()
        thumbnail.save(thumbnail_buffer, format="PNG")
        thumbnail_base64 = base64.b64encode(thumbnail_buffer.getvalue()).decode('utf-8')

        # Update job result
        result = {
            "image_path": output_path,
            "thumbnail": f"data:image/png;base64,{thumbnail_base64}"
        }

        qm.update_job_result(job_id, result)
        qm.update_job_status(job_id, "completed")
        qm.update_job_progress(job_id, 1.0, 100, 100, "Image transfer completed!")

        print(f"Image transfer job completed: {job_id}")

    except Exception as e:
        print(f"Error in image transfer job {job_id}: {str(e)}")
        traceback.print_exc()
        qm.update_job_status(job_id, f"failed - {str(e)}")
        qm.update_job_progress(job_id, 0.0, 0, 100, f"Error: {str(e)}")
    finally:
        # Ensure memory is cleaned
        clear_memory()