import base64
import io
import os
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from api import queue_manager as qm
from api import settings
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.hunyuan import encode_prompt_conds, vae_encode
from diffusers_helper.load_lora import load_lora
from diffusers_helper.memory import gpu, fake_diffusers_current_device, load_model_as_complete, unload_complete_models, move_model_to_device_with_memory_preservation
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.utils import crop_or_pad_yield_mask, resize_and_center_crop, write_PIL_image_with_png_info


def process_image_generation(job_id: str, job_data: dict, models: dict):
    """
    Process single image-to-image generation job using Hunyuan Video model for 1 frame
    Requires an input image to be provided
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
        if not input_image_b64:
            raise ValueError("Input image is required for image generation")

        # Decode base64 input image
        input_image = Image.open(io.BytesIO(base64.b64decode(input_image_b64))).convert('RGB')

        # Process input image
        input_image_np = np.array(input_image)
        h, w, c = input_image_np.shape
        height, width = find_nearest_bucket(h, w, resolution=640)
        input_image_np = resize_and_center_crop(
            input_image_np, target_width=width, target_height=height
        )

        # Convert to PIL for CLIP encoding
        # pil_input_image = Image.fromarray(input_image_np)

        # Clean GPU if not high_vram
        if not high_vram:
            qm.update_job_progress(job_id, 0.05, 5, 100, "Cleaning GPU memory...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        qm.update_job_progress(job_id, 0.1, 10, 100, "Encoding prompts...")
        
        # Prepare models for text encoding in low VRAM mode
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )

        if llama_vec is None or clip_l_pooler is None:
            raise ValueError("Failed to encode prompt")

        # Handle negative prompt
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
            if llama_vec_n is None or clip_l_pooler_n is None:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # Load LoRA if specified
        if lora_path:
            full_lora_path = os.path.join(settings.LORA_DIR, lora_path) if not os.path.isabs(lora_path) else lora_path
            if os.path.exists(full_lora_path):
                qm.update_job_progress(job_id, 0.15, 15, 100, f"Loading LoRA: {lora_path}")
                lora_dir, lora_name = os.path.split(full_lora_path)
                transformer = load_lora(transformer, Path(lora_dir), lora_name, lora_scale=lora_scale)
                # Ensure LoRA parameters are on GPU
                transformer = transformer.to(gpu)

        # Encode input image with CLIP vision
        qm.update_job_progress(job_id, 0.18, 18, 100, "Encoding input image...")
        if not high_vram:
            image_encoder = image_encoder.to(gpu)
        vision_hidden_states = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder).last_hidden_state.to(transformer.dtype)
        if not high_vram:
            image_encoder = image_encoder.cpu()
            torch.cuda.empty_cache()

        # Encode input image with VAE
        qm.update_job_progress(job_id, 0.2, 20, 100, "Encoding with VAE...")
        video_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        video_pt = video_pt.permute(2, 0, 1)[None, :, None]

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        latents = vae_encode(video_pt, vae)
        if not high_vram:
            vae = vae.cpu()
            torch.cuda.empty_cache()

        # Prepare latents
        qm.update_job_progress(job_id, 0.25, 25, 100, "Preparing latents...")
        latents = latents.to(torch.bfloat16).to(gpu)

        # Models are already on GPU from text encoding, no need to move again

        llama_vec = llama_vec.to(transformer.dtype).to(gpu)
        llama_vec_n = llama_vec_n.to(transformer.dtype).to(gpu)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype).to(gpu)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype).to(gpu)
        llama_attention_mask = llama_attention_mask.to(gpu)
        llama_attention_mask_n = llama_attention_mask_n.to(gpu)

        # Move transformer to GPU with memory preservation
        if not high_vram:
            unload_complete_models()  # Unload other models first
            if not next(transformer.parameters()).is_cuda:  # Only move if not already on GPU
                move_model_to_device_with_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation
                )
            else:
                print(f"Job {job_id}: Transformer already on GPU, skipping memory preservation move.")

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
                sampler='unipc',
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

        # Decode latents
        qm.update_job_progress(job_id, 0.9, 90, 100, "Decoding image...")

        if not high_vram:
            vae = vae.to(gpu)
            transformer = transformer.cpu()
            torch.cuda.empty_cache()

        generated = generated.to(torch.float16)

        with torch.no_grad():
            frames = vae.decode(generated / vae.config.scaling_factor, return_dict=False)[0]

        if not high_vram:
            vae = vae.cpu()
            torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()

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
        print(f"Error in image generation job {job_id}: {str(e)}")
        traceback.print_exc()
        qm.update_job_status(job_id, f"failed - {str(e)}")
        qm.update_job_progress(job_id, 0.0, 0, 100, f"Error: {str(e)}")


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
        # batch_size = data.get("batch_size", 4)
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
        target_pil = Image.fromarray(target_np)

        # Clean GPU if not high_vram
        if not high_vram:
            qm.update_job_progress(job_id, 0.15, 15, 100, "Cleaning GPU memory...")
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Encode prompts
        qm.update_job_progress(job_id, 0.2, 20, 100, "Encoding prompts...")
        
        # Prepare models for text encoding in low VRAM mode
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)
        
        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                negative_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )
            if llama_vec_n is None:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)

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
        vision_hidden_states = hf_clip_vision_encode(source_np, feature_extractor, image_encoder).last_hidden_state.to(transformer.dtype)
        if not high_vram:
            image_encoder = image_encoder.cpu()
            torch.cuda.empty_cache()

        # Encode target image with VAE
        qm.update_job_progress(job_id, 0.4, 40, 100, "Encoding target image...")
        video_pt = torch.from_numpy(target_np).float() / 127.5 - 1
        video_pt = video_pt.permute(2, 0, 1)[None, :, None]

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        latents = vae_encode(video_pt, vae)
        if not high_vram:
            vae = vae.cpu()
            torch.cuda.empty_cache()

        latents = latents.to(torch.bfloat16).to(gpu)

        # Add noise based on transfer strength
        noise = torch.randn_like(latents)
        latents = latents * (1 - transfer_strength) + noise * transfer_strength

        # Move transformer to GPU with memory preservation
        if not high_vram:
            unload_complete_models()  # Unload other models first
            if not next(transformer.parameters()).is_cuda:  # Only move if not already on GPU
                move_model_to_device_with_memory_preservation(
                    transformer,
                    target_device=gpu,
                    preserved_memory_gb=gpu_memory_preservation
                )
            else:
                print(f"Job {job_id}: Transformer already on GPU, skipping memory preservation move.")

        llama_vec = llama_vec.to(transformer.dtype).to(gpu)
        llama_vec_n = llama_vec_n.to(transformer.dtype).to(gpu)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype).to(gpu)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype).to(gpu)
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
                sampler='unipc',
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
                strength=transfer_strength,
                image_embeddings=vision_hidden_states,
                callback=progress_callback
            )

        # Decode
        qm.update_job_progress(job_id, 0.9, 90, 100, "Decoding result...")

        if not high_vram:
            vae = vae.to(gpu)
            transformer = transformer.cpu()
            torch.cuda.empty_cache()

        generated = generated.to(torch.float16)

        with torch.no_grad():
            frames = vae.decode(generated / vae.config.scaling_factor, return_dict=False)[0]

        if not high_vram:
            vae = vae.cpu()
            text_encoder = text_encoder.cpu()
            text_encoder_2 = text_encoder_2.cpu()
            torch.cuda.empty_cache()

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
