import os
import torch
import numpy as np
from pathlib import Path
import logging
from PIL import Image
# from PIL.PngImagePlugin import PngInfo
import traceback
import base64
import io
import einops

# Assuming models and tokenizers are loaded elsewhere and passed or accessed globally/via context
# This will be refined when creating models.py and integrating
# from diffusers import AutoencoderKLHunyuanVideo
# from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_encode,
    vae_decode_fake,
)

from diffusers_helper.utils import (
    save_bcthw_as_mp4,
    crop_or_pad_yield_mask,
    resize_and_center_crop,
    soft_append_bcthw,
)

from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import (
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
    fake_diffusers_current_device,
    unload_complete_models,
    load_model_as_complete,
)

from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Assuming queue_manager is available for status updates (relative import)
from . import queue_manager
from .queue_manager import update_job_progress
from . import settings
from diffusers_helper.load_lora import load_lora

# Define output folder using settings
outputs_folder = settings.OUTPUTS_DIR
# os.makedirs(outputs_folder, exist_ok=True)

# Determine VRAM mode - consider moving to settings.py or detecting dynamically
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f"Worker: Free VRAM {free_mem_gb} GB")
print(f"Worker: High-VRAM Mode: {high_vram}")


@torch.no_grad()
def worker(job: queue_manager.QueuedJob, models: dict):
    """
    Processes a single video generation job.
    Args:
        job (QueuedJob): The job object containing parameters.
        models (dict): A dictionary containing the loaded models and tokenizers.
                       Expected keys: 'vae', 'text_encoder', 'text_encoder_2',
                                      'image_encoder', 'transformer_base', 'transformer_f1',
                                      'tokenizer', 'tokenizer_2', 'feature_extractor'.
    """
    input_image_path = job.image_path
    prompt = job.prompt
    # n_prompt = job.n_prompt
    n_prompt = ""
    seed = job.seed
    total_second_length = job.video_length
    latent_window_size = 9
    steps = job.steps
    cfg = job.cfg
    gs = job.gs
    rs = job.rs
    gpu_memory_preservation = job.gpu_memory_preservation
    use_teacache = job.use_teacache
    mp4_crf = job.mp4_crf
    job_id = job.job_id
    lora_scale = job.lora_scale
    lora_path = job.lora_path
    original_exif = job.original_exif
    # --- Get sampling mode and transformer model from job ---
    sampling_mode = job.sampling_mode
    transformer_model_name = job.transformer_model
    print(f"Job {job_id}: Sampling Mode='{sampling_mode}', Transformer Model='{transformer_model_name}'")

    thumbnail_path = None

    # Update job status to processing, including the thumbnail path (will be updated again if thumbnail generated)
    # We update here initially in case thumbnail generation fails later
    queue_manager.update_job_status(job_id, "processing", thumbnail=thumbnail_path)

    # Load models from the dictionary
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    image_encoder = models["image_encoder"]
    # --- Select the correct transformer model ---
    if transformer_model_name == "f1":
        transformer = models["transformer_f1"]
        print(f"Job {job_id}: Using F1 Transformer model.")
    else:
        transformer = models["transformer_base"]
        print(f"Job {job_id}: Using Base Transformer model.")
    # --- End Transformer Selection ---
    tokenizer = models["tokenizer"]
    tokenizer_2 = models["tokenizer_2"]
    feature_extractor = models["feature_extractor"]

    # Progress update function that calls queue_manager
    def update_progress(step_info: str, percentage: float = 0.0, current_step: int = 0, total_steps: int = 0):
        """Updates progress in the console and via queue_manager."""
        print(f"Job {job_id} Progress: {step_info} ({percentage:.1f}%)")
        try:
            # Use the new function to update progress details in the queue manager
            update_job_progress(
                job_id=job_id,
                progress=percentage,
                step=current_step,
                total=total_steps if total_steps > 0 else steps,
                info=step_info
            )
        except Exception as e:
            print(f"Error updating job progress for {job_id}: {e}")
            # Decide if this error should halt the process or just be logged

    update_progress("Starting ...", 0, 0, steps)

    # Initialize history_pixels before the main try block
    history_pixels = None

    try:
        # Load input image try-except
        try:
            pil_input_image = Image.open(input_image_path)
            # logging.info(f"[Job {job_id}] Exif in input_image after open: {pil_input_image.info.get('exif') is not None}")

            # --- Thumbnail Generation (Moved Here) ---
            try:
                # Generate thumbnail from the loaded input image (pil_input_image)
                thumb_size = (128, 128)
                thumb_img = pil_input_image.copy()
                thumb_img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                thumbnail_filename = f"thumb_{job_id}.jpg"
                # Use previously initialized thumbnail_path variable
                thumbnail_path = os.path.join(settings.TEMP_QUEUE_IMAGES_DIR, thumbnail_filename)
                thumb_img.save(thumbnail_path, "JPEG", quality=85)
                print(f"Job {job_id}: Thumbnail saved to {thumbnail_path}")
                # Update job status again with the actual thumbnail path
                queue_manager.update_job_status(job_id, "processing", thumbnail=thumbnail_path)
            except Exception as thumb_e:
                print(f"Job {job_id}: Warning - Failed to generate thumbnail: {thumb_e}")
                thumbnail_path = None

            # input_image = np.array(pil_input_image)

        except FileNotFoundError:
            print(f"Error: Input image not found at {input_image_path}")
            queue_manager.update_job_status(job_id, "failed - image not found")
            return
        except Exception as e:
            print(f"Error loading image {input_image_path}: {e}")
            queue_manager.update_job_status(job_id, f"failed - image load error: {e}")
            return

        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        # Clean GPU if not high_vram
        if not high_vram:
            update_progress("Cleaning GPU memory...", 1, 0, steps)
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        update_progress("Text encoding ...", 5, 0, steps)
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(
            prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
        )

        # Check if encoding returned None
        if llama_vec is None or clip_l_pooler is None:
            print(f"Error: Failed to encode positive prompt for job {job_id}")
            queue_manager.update_job_status(job_id, "failed - prompt encoding error")
            return

        if cfg == 1:
            # Ensure llama_vec is valid before creating zeros_like
            if llama_vec is None or clip_l_pooler is None:
                print(
                    f"Error: Cannot create negative embeddings because positive embeddings are None for job {job_id}"
                )
                queue_manager.update_job_status(
                    job_id, "failed - prompt encoding error"
                )
                return
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(
                llama_vec
            ), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
            )

            # Check if encoding returned None for negative prompt
            if llama_vec_n is None or clip_l_pooler_n is None:
                print(
                    f"Warning: Failed to encode negative prompt for job {job_id}. Using zeros."
                )
                # Fallback to zeros if negative encoding fails but cfg != 1
                if (
                    llama_vec is None or clip_l_pooler is None
                ):
                    print(
                        f"Error: Cannot create negative embeddings because positive embeddings are None for job {job_id}"
                    )
                    queue_manager.update_job_status(
                        job_id, "failed - prompt encoding error"
                    )
                    return
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(
                    llama_vec
                ), torch.zeros_like(clip_l_pooler)

        # Ensure embeddings are not None before padding
        if llama_vec is None or llama_vec_n is None:
            print(f"Error: Embeddings are None before padding for job {job_id}")
            queue_manager.update_job_status(job_id, "failed - prompt encoding error")
            return

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
            llama_vec_n, length=512
        )

        # --- LoRA Loading (moved here, after text encoding, before image processing) ---
        # Construct full path if lora_path is provided (assumed to be filename from /loras endpoint)
        full_lora_path = None
        if lora_path:
            # Check if lora_path is already an absolute path (optional, for flexibility)
            if os.path.isabs(lora_path):
                full_lora_path = lora_path
            else:
                # Assume it's a filename and join with LORA_DIR
                full_lora_path = os.path.join(settings.LORA_DIR, lora_path)

        if full_lora_path and os.path.exists(full_lora_path):
            print(f"Job {job_id}: Loading LoRA from: {full_lora_path} with scale {lora_scale}")
            update_progress(f"Loading LoRA '{lora_path}' (scale={lora_scale})...", 11, 0, steps)
            try:
                # load_lora expects directory and filename separately
                lora_dir, lora_name = os.path.split(full_lora_path)
                # transformer 変数を更新する
                transformer = load_lora(transformer, Path(lora_dir), lora_name, lora_scale=lora_scale)
                print(f"Job {job_id}: LoRA loaded successfully.")
            except Exception as e:
                print(f"Job {job_id}: Error loading LoRA: {e}")
                # queue_manager.update_job_status(job_id, f"failed - LoRA load error: {e}")
                # return
        elif lora_path:
            print(f"Job {job_id}: Warning - LoRA path '{lora_path}' specified but file not found at '{full_lora_path}'.")
        else:
            print(f"Job {job_id}: No LoRA path specified, skipping LoRA loading.")
        # --- End LoRA Loading ---

        # Processing input image (Convert to numpy array here if needed for processing)
        input_image = np.array(pil_input_image)
        update_progress("Image processing ...", 12, 0, steps)
        input_image = np.squeeze(input_image)
        if input_image.ndim != 3 or input_image.shape[2] != 3:
            print(f"Error: Invalid image shape {input_image.shape} for job {job_id}")
            queue_manager.update_job_status(job_id, "failed - invalid image shape")
            return

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(
            H, W, resolution=640
        )
        input_image_np = resize_and_center_crop(
            input_image, target_width=width, target_height=height
        )

        # Save a copy of the processed input image (optional, but good for reference)
        # Change filename to .jpg
        processed_input_image_path = os.path.join(outputs_folder, f"{job_id}_input.jpg")
        # Create PIL image from numpy array for saving
        processed_pil_image = Image.fromarray(input_image_np)

        # Prepare save arguments for JPEG with Exif
        save_kwargs = {
            "format": "JPEG",
            "quality": 70,
        }
        # Add exif data if it exists (retrieved from the job object)
        if original_exif:
            save_kwargs["exif"] = original_exif
            logging.info(f"[Job {job_id}] Attempting to save processed input with Exif data.")
        else:
            logging.info(f"[Job {job_id}] No Exif data found in job object for processed input.")

        # Save processed input image as JPEG with or without Exif
        processed_pil_image.save(processed_input_image_path, **save_kwargs)
        # logging.info(f"[Job {job_id}] Saved processed input image to {processed_input_image_path} (JPEG)")

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        update_progress("VAE encoding ...", 15, 0, steps)
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision encoding
        update_progress("CLIP Vision encoding ...", 20, 0, steps)
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(
            input_image_np, feature_extractor, image_encoder
        )
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype conversion
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(
            transformer.dtype
        )

        # Sampling Logic based on sampling_mode
        update_progress("Start sampling ...", 25, 0, steps)
        rnd = torch.Generator("cpu").manual_seed(seed)

        # ==============================================================
        # === Forward Sampling Mode (like demo_gradio_f1.py) ===
        # ==============================================================
        if sampling_mode == "forward":
            print(f"Job {job_id}: Entering FORWARD sampling mode.")
            history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
            # history_pixels initialized before try block

            history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
            total_generated_latent_frames = 1

            sampling_step_count = total_latent_sections
            current_sampling_step = 0

            for section_index in range(total_latent_sections):
                current_sampling_step += 1
                section_progress_start = 25 + (current_sampling_step - 1) * (70 / sampling_step_count)
                section_progress_end = 25 + current_sampling_step * (70 / sampling_step_count)

                # Check for cancellation signal
                current_job_status_section_start = queue_manager.get_job_by_id(job_id)
                if current_job_status_section_start and current_job_status_section_start.status == "cancelled":
                    print(f"Job {job_id} cancellation detected at start of forward section {current_sampling_step}.")
                    # Update status and exit worker function
                    queue_manager.update_job_status(job_id, "cancelled")
                    return

                print(f'Job {job_id}: Forward section_index = {section_index}, total_latent_sections = {total_latent_sections}')
                update_progress(
                    f"Sampling forward section {current_sampling_step}/{sampling_step_count}",
                    section_progress_start, 0, steps
                )

                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                # K-Diffusion Sampling Callback (Forward Mode)
                def callback_forward(d):
                    current_cb_step = d['i'] + 1
                    total_cb_steps = steps
                    section_progress_fraction = current_cb_step / total_cb_steps
                    overall_sampling_progress = section_progress_fraction * (70 / sampling_step_count)
                    overall_percentage = section_progress_start + overall_sampling_progress

                    hint = f'Sampling forward section {current_sampling_step}/{sampling_step_count} - Step {current_cb_step}/{total_cb_steps}'
                    print(f"Job {job_id} Progress: {hint} ({overall_percentage:.1f}%)")
                    update_job_progress(job_id=job_id, progress=overall_percentage, step=current_cb_step, total=total_cb_steps, info=hint)

                    # Preview Generation (same as reverse mode callback)
                    try:
                        if current_cb_step % 2 == 0 or current_cb_step == total_cb_steps:
                            preview_latent = d['denoised']
                            preview_tensor = vae_decode_fake(preview_latent)
                            preview_np = (preview_tensor * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                            preview_np_rearranged = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')
                            preview_image = Image.fromarray(preview_np_rearranged)
                            buffer = io.BytesIO()
                            preview_image.save(buffer, format="JPEG", quality=75)
                            buffer.seek(0)
                            preview_base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            preview_base64_string = f"data:image/jpeg;base64,{preview_base64_data}"
                            queue_manager.update_current_preview(job_id, preview_base64_string)
                    except Exception as preview_e:
                        logging.warning(f"Job {job_id}: Error generating preview at forward step {current_cb_step}: {preview_e}")

                    # Cancellation Check
                    current_job_status_inner = queue_manager.get_job_by_id(job_id)
                    if current_job_status_inner and current_job_status_inner.status == "cancelled":
                        print(f"Job {job_id} cancelled during forward sampling step {current_cb_step}.")
                        raise InterruptedError("Job cancelled")

                # Prepare arguments for sample_hunyuan (Forward Mode)
                indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

                clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
                clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

                try:
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=latent_window_size * 4 - 3,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
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
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback_forward,
                    )
                except InterruptedError:
                    print(f"Job {job_id}: Cancellation detected during forward sampling.")
                    # Update status and exit worker function
                    queue_manager.update_job_status(job_id, "cancelled")
                    return

                # Update history (Forward Mode)
                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

                # Decode and append frames (Forward Mode)
                update_progress(
                    f"VAE decoding forward section {current_sampling_step}/{sampling_step_count}",
                    section_progress_end - 1, 0, steps
                )
                if not high_vram:
                    unload_complete_models()
                    load_model_as_complete(vae, target_device=gpu)

                real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

                if history_pixels is None:
                    # This should only happen on the very first section, but the logic is slightly different
                    # demo_gradio_f1 starts history_pixels after the first sampling loop completes.
                    # We decode the *entire* history up to this point.
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    # Decode the newly generated section and append
                    section_latent_frames = latent_window_size * 2
                    overlapped_frames = latent_window_size * 4 - 3

                    # Decode only the relevant part of the history for the current section
                    # We need the last `section_latent_frames` from the *generated* latents part
                    current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

                print(f'Job {job_id}: Decoded forward. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                # Save intermediate result (optional, can be useful for debugging)
                # intermediate_filename = os.path.join(outputs_folder, f'{job_id}_forward_part_{current_sampling_step}.mp4')
                # save_bcthw_as_mp4(history_pixels, intermediate_filename, crf=mp4_crf, fps=30)

        # ==============================================================
        # === Reverse Sampling Mode (Original Logic) ===
        # ==============================================================
        else:
            print(f"Job {job_id}: Entering REVERSE sampling mode (default).")
            # Correct initialization for reverse mode
            history_latents = torch.zeros(
                size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32
            ).cpu()
            # history_pixels initialized before try block
            total_generated_latent_frames = 0

            latent_paddings = reversed(range(total_latent_sections))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            else:
                latent_paddings = list(latent_paddings)

            sampling_step_count = len(latent_paddings)
            current_sampling_step = 0

            for latent_padding in latent_paddings:
                current_sampling_step += 1
                section_progress_start = 25 + (current_sampling_step - 1) * (70 / sampling_step_count)
                section_progress_end = 25 + current_sampling_step * (70 / sampling_step_count)

                # Check for cancellation signal at the beginning of each section
                current_job_status_section_start = queue_manager.get_job_by_id(job_id)
                if current_job_status_section_start and current_job_status_section_start.status == "cancelled":
                    print(f"Job {job_id} cancellation detected at start of reverse section {current_sampling_step}.")
                    # Update status and exit worker function
                    queue_manager.update_job_status(job_id, "cancelled")
                    return

                is_last_section = latent_padding == 0
                latent_padding_size = latent_padding * latent_window_size

                print(
                    f"Job {job_id}: latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}"
                )
                # Update progress for the start of the section
                update_progress(
                    f"Sampling reverse section {current_sampling_step}/{sampling_step_count}",
                    section_progress_start,
                    0,
                    steps
                )

                indices = torch.arange(
                    0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
                ).unsqueeze(0)
                (
                    clean_latent_indices_pre,
                    blank_indices,
                    latent_indices,
                    clean_latent_indices_post,
                    clean_latent_2x_indices,
                    clean_latent_4x_indices,
                ) = indices.split(
                    [1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1
                )
                clean_latent_indices = torch.cat(
                    [clean_latent_indices_pre, clean_latent_indices_post], dim=1
                )

                clean_latents_pre = start_latent.to(history_latents)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[
                    :, :, : 1 + 2 + 16, :, :
                ].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(
                        transformer,
                        target_device=gpu,
                        preserved_memory_gb=gpu_memory_preservation,
                    )

                if use_teacache:
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                # K-Diffusion Sampling Callback (Reverse Mode)
                def callback(d):
                    # --- Progress Update ---
                    current_cb_step = d['i'] + 1
                    total_cb_steps = steps

                    # Calculate overall progress percentage
                    section_progress_fraction = current_cb_step / total_cb_steps
                    overall_sampling_progress = section_progress_fraction * (70 / sampling_step_count)
                    overall_percentage = section_progress_start + overall_sampling_progress

                    hint = f'Sampling reverse section {current_sampling_step}/{sampling_step_count} - Step {current_cb_step}/{total_cb_steps}'
                    print(f"Job {job_id} Progress: {hint} ({overall_percentage:.1f}%)")

                    # Update progress via queue_manager
                    update_job_progress(
                        job_id=job_id,
                        progress=overall_percentage,
                        step=current_cb_step,
                        total=total_cb_steps,
                        info=hint
                    )

                    # --- Preview Generation ---
                    try:
                        if current_cb_step % 2 == 0 or current_cb_step == total_cb_steps:
                            preview_latent = d['denoised']
                            preview_tensor = vae_decode_fake(preview_latent)  # Use vae_decode_fake

                            preview_np = (preview_tensor * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                            preview_np_rearranged = einops.rearrange(preview_np, 'b c t h w -> (b h) (t w) c')

                            preview_image = Image.fromarray(preview_np_rearranged)

                            buffer = io.BytesIO()
                            preview_image.save(buffer, format="JPEG", quality=75)
                            buffer.seek(0)

                            preview_base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                            preview_base64_string = f"data:image/jpeg;base64,{preview_base64_data}"

                            queue_manager.update_current_preview(job_id, preview_base64_string)
                    except Exception as preview_e:
                        logging.warning(f"Job {job_id}: Error generating preview at reverse step {current_cb_step}: {preview_e}")

                    # --- Cancellation Check ---
                    current_job_status_inner = queue_manager.get_job_by_id(job_id)
                    if current_job_status_inner and current_job_status_inner.status == "cancelled":
                        print(f"Job {job_id} cancelled during reverse sampling step {current_cb_step}.")
                        raise InterruptedError("Job cancelled")

                try:
                    num_frames = latent_window_size * 4 - 3
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
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
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback,
                    )
                except InterruptedError:
                    print(f"Job {job_id}: Cancellation detected during reverse sampling.")
                    # Update status and exit worker function
                    queue_manager.update_job_status(job_id, "cancelled")
                    return

                # Update history (Reverse Mode)
                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                # Decode and append frames (Reverse Mode)
                update_progress(
                    f"VAE decoding reverse section {current_sampling_step}/{sampling_step_count}",
                    section_progress_end - 1, 0, steps
                )
                if not high_vram:
                    unload_complete_models()
                    load_model_as_complete(vae, target_device=gpu)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3

                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                print(f'Job {job_id}: Decoded reverse. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                # Save intermediate result (optional)
                # intermediate_filename = os.path.join(outputs_folder, f'{job_id}_reverse_part_{current_sampling_step}.mp4')
                # save_bcthw_as_mp4(history_pixels, intermediate_filename, crf=mp4_crf, fps=30)

                if is_last_section:
                    break

        # ==============================================================
        # === Final Save (Common to both modes) ===
        # ==============================================================
        update_progress("Saving final video...", 98, 0, steps)
        final_filename = os.path.join(outputs_folder, f"{job_id}.mp4")
        if history_pixels is not None:
            save_bcthw_as_mp4(history_pixels, final_filename, crf=mp4_crf, fps=30)
            # Update job status and final progress
            update_progress("Finished", 100, steps, steps)  # Mark 100% progress
            queue_manager.update_job_status(job_id, "completed")
            print(f"Job {job_id} completed successfully. Output: {final_filename}")
        else:
            # This should not happen if sampling ran, but handle defensively
            print(f"Error: history_pixels is None after sampling for job {job_id}. Cannot save video.")
            queue_manager.update_job_status(job_id, "failed - internal error (no pixels)")

    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")
        traceback.print_exc()
        # Update status to failed, keep last known progress but update status and info.
        last_progress = queue_manager.get_job_by_id(job_id)  # Get current state before updating status
        last_perc = last_progress.progress if last_progress else 0
        last_step = last_progress.progress_step if last_progress else 0
        last_total = last_progress.progress_total if last_progress else steps
        fail_info = f"Failed: {type(e).__name__}"
        update_job_progress(job_id, last_perc, last_step, last_total, fail_info)
        queue_manager.update_job_status(job_id, f"failed - {type(e).__name__}")
        print(f"Job {job_id} failed: {fail_info}")

    finally:
        # Final GPU cleanup (optional, depends on worker lifecycle)
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        # Clear preview data from memory
        queue_manager.clear_current_preview(job_id)
        print(f"Worker finished for job {job_id}")


# Example usage (for testing purposes, would be called by a background task runner)
if __name__ == "__main__":
    print("Worker module loaded. Contains the 'worker' function.")
    # Add test code here if needed
    pass
