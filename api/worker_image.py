import os
import torch
import numpy as np
from PIL import Image
import io
import base64
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import gc
import asyncio

from diffusers_helper.utils import save_image
from diffusers_helper.memory import gpu
from diffusers_helper.clip_vision import preprocess_image_to_embed
from diffusers_helper.dit_common import is_teacache_folder_exist, TeaCacheTestBackend
from diffusers_helper.load_lora import update_lora_weight_to_transformer
from diffusers_helper.k_diffusion.wrapper import HyVideoSampler
from diffusers_helper.k_diffusion.uni_pc_fm import UniPCMultistepScheduler

from api.queue_manager import QueueManager
from api.settings import get_settings
import torch.nn.functional as F


def encode_cropped_prompt(prompt, negative_prompt, tokenizer, tokenizer_2, text_encoder, text_encoder_2):
    """Encode text prompts using dual text encoders"""
    device = text_encoder.device if hasattr(text_encoder, 'device') else gpu
    
    # Tokenize prompts
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    
    text_input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    
    # Encode with first text encoder
    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
    )[0]
    
    # Tokenize for second encoder
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    
    # Encode with second text encoder
    text_encoder_2_out = text_encoder_2(
        text_inputs_2.input_ids.to(device),
        output_hidden_states=True,
    )
    
    pooled_prompt_embeds = text_encoder_2_out[0]
    prompt_embeds_2 = text_encoder_2_out.hidden_states[-2]
    
    # Process negative prompt
    if negative_prompt:
        # Similar process for negative prompt
        negative_text_inputs = tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        
        negative_prompt_embeds = text_encoder(
            negative_text_inputs.input_ids.to(device),
            attention_mask=negative_text_inputs.attention_mask.to(device),
        )[0]
        
        negative_text_inputs_2 = tokenizer_2(
            negative_prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        
        negative_text_encoder_2_out = text_encoder_2(
            negative_text_inputs_2.input_ids.to(device),
            output_hidden_states=True,
        )
        
        negative_pooled_prompt_embeds = negative_text_encoder_2_out[0]
        negative_prompt_embeds_2 = negative_text_encoder_2_out.hidden_states[-2]
    else:
        # Use empty embeddings for negative prompt
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_prompt_embeds_2 = torch.zeros_like(prompt_embeds_2)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
    
    # Concatenate embeddings
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    negative_prompt_embeds = torch.cat([negative_prompt_embeds, negative_prompt_embeds_2], dim=-1)
    
    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds


def prepare_latents(
    shape: Tuple[int, int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Prepare initial latents for image generation"""
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    return latents


async def process_image_generation(job_id: str, job_data: dict, models: dict):
    """
    Process single image generation job
    """
    queue_manager = QueueManager()
    
    try:
        print(f"Starting image generation job: {job_id}")
        queue_manager.update_job_status(job_id, "processing")
        queue_manager.update_job_progress(job_id, 0.0, 0, 100, "Initializing image generation...")
        
        # Extract parameters
        data = job_data.get("data", {})
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        seed = data.get("seed", -1)
        steps = data.get("steps", 30)
        cfg = data.get("cfg", 1.0)
        width = data.get("width", 1216)
        height = data.get("height", 704)
        lora_paths = data.get("lora_paths", [])
        lora_scales = data.get("lora_scales", [])
        
        # Set random seed
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)
        generator = torch.Generator(device=gpu).manual_seed(seed)
        
        # Get models
        vae = models["vae"]
        text_encoder = models["text_encoder"]
        text_encoder_2 = models["text_encoder_2"]
        tokenizer = models["tokenizer"]
        tokenizer_2 = models["tokenizer_2"]
        transformer = models["transformer_base"]
        high_vram = models["high_vram"]
        
        # Load LoRA if specified
        if lora_paths and len(lora_paths) > 0:
            print(f"Loading LoRA: {lora_paths}")
            for lora_path, lora_scale in zip(lora_paths, lora_scales or [1.0] * len(lora_paths)):
                update_lora_weight_to_transformer(transformer, lora_path, lora_scale)
        
        # Move models to GPU if needed (low VRAM mode)
        if not high_vram:
            text_encoder_2.to(gpu)
            vae.to(gpu)
        
        # Encode prompts
        queue_manager.update_job_progress(job_id, 0.1, 10, 100, "Encoding prompts...")
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
            encode_cropped_prompt(
                prompt, negative_prompt,
                tokenizer, tokenizer_2,
                text_encoder, text_encoder_2
            )
        
        # Prepare latents
        batch_size = 1
        num_channels_latents = 16
        latent_height = height // 8
        latent_width = width // 8
        
        latents_shape = (
            batch_size,
            num_channels_latents,
            1,  # Single frame for image
            latent_height,
            latent_width
        )
        
        latents = prepare_latents(
            latents_shape,
            transformer.dtype,
            gpu,
            generator
        )
        
        # Setup sampler
        queue_manager.update_job_progress(job_id, 0.2, 20, 100, "Setting up sampler...")
        sampler = HyVideoSampler(
            transformer,
            latents.shape,
            prompt_embeds.shape[1],
            prompt_embeds.dtype,
            pooled_prompt_embeds.dtype,
            False,  # low_gpu_memory
            device=gpu
        )
        
        # Setup scheduler
        scheduler = UniPCMultistepScheduler()
        
        # Sampling parameters
        sampler.model_kwargs = {
            "encoder_hidden_states": prompt_embeds,
            "guidance_scale": cfg,
            "pooled_encoder_hidden_states": pooled_prompt_embeds,
            "negative_encoder_hidden_states": negative_prompt_embeds if cfg > 1.0 else None,
            "negative_pooled_encoder_hidden_states": negative_pooled_prompt_embeds if cfg > 1.0 else None,
        }
        
        # Generate image
        queue_manager.update_job_progress(job_id, 0.3, 30, 100, "Generating image...")
        
        # Use TeaCache if available
        if is_teacache_folder_exist():
            sampler.model_kwargs["teacache_backend"] = TeaCacheTestBackend()
        
        # Run diffusion steps
        for i in range(steps):
            # Update progress
            progress = 0.3 + (0.5 * (i / steps))
            queue_manager.update_job_progress(
                job_id, progress, 30 + int(50 * (i / steps)), 100,
                f"Diffusion step {i+1}/{steps}"
            )
            
            # Perform single diffusion step
            timestep = scheduler.timesteps[i]
            noise_pred = sampler.model(latents, timestep, **sampler.model_kwargs)
            latents = scheduler.step(noise_pred, timestep, latents).prev_sample
        
        # Decode latents
        queue_manager.update_job_progress(job_id, 0.8, 80, 100, "Decoding image...")
        
        # Ensure VAE is on GPU
        if not high_vram:
            vae.to(gpu)
        
        # Decode
        with torch.no_grad():
            image_tensor = vae.decode(latents[:, :, 0, :, :] / vae.config.scaling_factor, return_dict=False)[0]
        
        # Convert to PIL Image
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
        image = Image.fromarray((image_tensor[0] * 255).astype(np.uint8))
        
        # Save image
        queue_manager.update_job_progress(job_id, 0.9, 90, 100, "Saving image...")
        
        output_dir = "outputs/images"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{job_id}.png")
        
        # Prepare metadata
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "model": "FramePack-I2V",
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save with metadata
        save_image(image, output_path, metadata)
        
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
        
        queue_manager.update_job_result(job_id, result)
        queue_manager.update_job_status(job_id, "completed")
        queue_manager.update_job_progress(job_id, 1.0, 100, 100, "Image generation completed!")
        
        # Cleanup
        if not high_vram:
            text_encoder_2.cpu()
            vae.cpu()
            torch.cuda.empty_cache()
        
        print(f"Image generation job completed: {job_id}")
        
    except Exception as e:
        print(f"Error in image generation job {job_id}: {str(e)}")
        queue_manager.update_job_status(job_id, "failed", str(e))
        queue_manager.update_job_progress(job_id, 0.0, 0, 100, f"Error: {str(e)}")
        
        # Cleanup on error
        if not high_vram:
            if 'text_encoder_2' in locals():
                text_encoder_2.cpu()
            if 'vae' in locals():
                vae.cpu()
            torch.cuda.empty_cache()


async def process_batch_images(job_id: str, job_data: dict, models: dict):
    """
    Process batch image generation job
    """
    # TODO: Implement batch processing
    # This will process multiple prompts in parallel
    pass


async def process_image_transfer(job_id: str, job_data: dict, models: dict):
    """
    Process image transfer (kisekaeichi) job
    """
    # TODO: Implement image transfer functionality
    # This will transfer style from source to target image
    pass