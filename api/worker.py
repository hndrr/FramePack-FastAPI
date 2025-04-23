import os
import torch
import numpy as np
from pathlib import Path  # 追加
from PIL import Image  # Removed ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import traceback

# Assuming models and tokenizers are loaded elsewhere and passed or accessed globally/via context
# This will be refined when creating models.py and integrating
# from diffusers import AutoencoderKLHunyuanVideo
# from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.hunyuan import (
    encode_prompt_conds,
    vae_decode,
    vae_encode,
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
from .queue_manager import update_job_progress  # Import the specific function
from . import settings  # Import settings to get OUTPUTS_DIR
from diffusers_helper.load_lora import load_lora  # 追加

# Define output folder using settings
outputs_folder = settings.OUTPUTS_DIR
# os.makedirs(outputs_folder, exist_ok=True) # Directory creation handled in settings.py

# Determine VRAM mode - consider moving to settings.py or detecting dynamically
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60  # Threshold might need adjustment
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
                                      'image_encoder', 'transformer', 'tokenizer',
                                      'tokenizer_2', 'feature_extractor'.
    """
    input_image_path = job.image_path
    prompt = job.prompt
    # n_prompt = job.n_prompt  # Assuming negative prompt might be added to QueuedJob
    n_prompt = ""  # Default negative prompt
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
    lora_scale = job.lora_scale  # 追加: lora_scale をジョブから取得
    lora_path = job.lora_path    # 追加: lora_path をジョブから取得

    # Update job status to processing
    queue_manager.update_job_status(job_id, "processing")

    # Load models from the dictionary
    vae = models["vae"]
    text_encoder = models["text_encoder"]
    text_encoder_2 = models["text_encoder_2"]
    image_encoder = models["image_encoder"]
    transformer = models["transformer"]
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
                total=total_steps if total_steps > 0 else steps,  # Use overall steps if section total is 0
                info=step_info
            )
        except Exception as e:
            print(f"Error updating job progress for {job_id}: {e}")
            # Decide if this error should halt the process or just be logged

    update_progress("Starting ...", 0, 0, steps)  # Initial progress

    try:
        # Load input image
        try:
            input_image = Image.open(input_image_path)
            input_image = np.array(input_image)
        except FileNotFoundError:
            print(f"Error: Input image not found at {input_image_path}")
            queue_manager.update_job_status(job_id, "failed - image not found")
            return
        except Exception as e:
            print(f"Error loading image {input_image_path}: {e}")
            queue_manager.update_job_status(job_id, f"failed - image load error: {e}")
            return  # Added return here
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        # Clean GPU if not high_vram
        if not high_vram:
            update_progress("Cleaning GPU memory...", 1, 0, steps)  # Small progress increment
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
                ):  # Should not happen due to earlier check, but safety first
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
        # --- LoRA Loading ---
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
            update_progress(f"Loading LoRA '{lora_path}' (scale={lora_scale})...", 11, 0, steps)  # Progress update with filename
            try:
                # load_lora expects directory and filename separately
                lora_dir, lora_name = os.path.split(full_lora_path)
                # transformer 変数を更新する
                transformer = load_lora(transformer, Path(lora_dir), lora_name, lora_scale=lora_scale)
                print(f"Job {job_id}: LoRA loaded successfully.")
            except Exception as e:
                print(f"Job {job_id}: Error loading LoRA: {e}")
                # LoRAロード失敗時の処理 (例: ログ出力して続行、ジョブを失敗させるなど)
                # queue_manager.update_job_status(job_id, f"failed - LoRA load error: {e}")
                # return
        elif lora_path:  # Only print warning if lora_path was provided but file not found
            print(f"Job {job_id}: Warning - LoRA path '{lora_path}' specified but file not found at '{full_lora_path}'.")
        else:
            print(f"Job {job_id}: No LoRA path specified, skipping LoRA loading.")
        # --- End LoRA Loading ---

        # Processing input image
        update_progress("Image processing ...", 12, 0, steps)  # Progress update (percentage adjusted)
        input_image = np.squeeze(input_image)  # Ensure 3D
        if input_image.ndim != 3 or input_image.shape[2] != 3:
            print(f"Error: Invalid image shape {input_image.shape} for job {job_id}")
            queue_manager.update_job_status(job_id, "failed - invalid image shape")
            return

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(
            H, W, resolution=640
        )  # Assuming default resolution
        input_image_np = resize_and_center_crop(
            input_image, target_width=width, target_height=height
        )

        # Save a copy of the processed input image (optional, but good for reference)
        processed_input_image_path = os.path.join(outputs_folder, f"{job_id}_input.png")
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        Image.fromarray(input_image_np).save(
            processed_input_image_path, pnginfo=metadata
        )

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

        # Sampling
        update_progress("Start sampling ...", 25, 0, steps)
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
            size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32
        ).cpu()
        history_pixels = None
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
            section_progress_start = 25 + (current_sampling_step - 1) * (
                70 / sampling_step_count
            )
            section_progress_end = 25 + current_sampling_step * (
                70 / sampling_step_count
            )

            # Check for cancellation signal at the beginning of each section
            current_job_status_section_start = queue_manager.get_job_by_id(job_id)
            if current_job_status_section_start and current_job_status_section_start.status == "cancelled":
                print(f"Job {job_id} cancellation detected at start of section {current_sampling_step}.")
                return

            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(
                f"Job {job_id}: latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}"
            )
            # Update progress for the start of the section
            update_progress(
                f"Sampling section {current_sampling_step}/{sampling_step_count}",
                section_progress_start,
                0,  # Step count resets for the section's callback
                steps  # Total steps for this section remains the overall steps parameter
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

            # K-Diffusion Sampling Callback
            def callback(d):
                current_cb_step = d['i'] + 1  # 1-based step for the current section
                total_cb_steps = steps  # Total steps for this section

                # Calculate overall progress percentage
                # Sampling spans from 25% to 95% (70% total)
                section_progress_fraction = current_cb_step / total_cb_steps
                overall_sampling_progress = section_progress_fraction * (70 / sampling_step_count)
                overall_percentage = section_progress_start + overall_sampling_progress

                hint = f'Sampling section {current_sampling_step}/{sampling_step_count} - Step {current_cb_step}/{total_cb_steps}'
                print(f"Job {job_id} Progress: {hint} ({overall_percentage:.1f}%)")

                # Update progress via queue_manager
                update_job_progress(
                    job_id=job_id,
                    progress=overall_percentage,
                    step=current_cb_step,
                    total=total_cb_steps,
                    info=hint
                )

                # Check for cancellation signal within callback
                current_job_status_inner = queue_manager.get_job_by_id(job_id)  # Use the function that reads the file
                if current_job_status_inner and current_job_status_inner.status == "cancelled":
                    # Use current_cb_step which is defined in this scope
                    print(f"Job {job_id} cancelled during sampling step {current_cb_step}.")
                    raise InterruptedError(
                        "Job cancelled"
                    )  # Raise exception to stop sampling

            try:
                # args_to_check = { ... }
                # none_args = [name for name, val in args_to_check.items() if val is None]
                # if none_args:
                #     error_msg = f"Error: The following arguments are None before calling sample_hunyuan: {', '.join(none_args)} for job {job_id}"
                #     print(error_msg)
                #     queue_manager.update_job_status(job_id, "failed - internal error (None arg)")
                #     return # Stop processing this section

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
                    # seed=seed + current_sampling_step,
                    # positive_image_encoder_hidden_states=image_encoder_last_hidden_state,
                    # negative_image_encoder_hidden_states=torch.zeros_like(image_encoder_last_hidden_state),
                )
            except InterruptedError:
                # Job was cancelled during sampling via callback
                return  # Exit worker function

            # Update history
            # Update total generated frames *before* updating history (moved from L446)
            total_generated_latent_frames += int(generated_latents.shape[2])
            # Update history by concatenating (like demo_gradio.py L557)
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            # Decode and append frames
            update_progress(
                f"VAE decoding section {current_sampling_step}/{sampling_step_count}",
                section_progress_end - 1,  # Approximate percentage
                0,  # Reset step count for this phase
                steps  # Use overall steps as total for this phase marker
            )
            if not high_vram:
                unload_complete_models()
                load_model_as_complete(vae, target_device=gpu)

            # Use the full history for decoding and appending (like demo_gradio.py L562-571)
            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                # Decode the entire history for the first section
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # Calculate frames for current section and overlap
                # Note: demo_gradio.py L567 seems to have a potential off-by-one or logic mismatch
                # compared to L553/L555. Using the logic from demo_gradio.py L567 & L570 for now.
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3  # Same calculation as demo_gradio.py L568

                # Decode only the relevant part of the history for the current section
                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                # Append using the calculated overlap (like demo_gradio.py L571)
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            # total_generated_latent_frames update moved before history_latents update (see L419 block)

            # Save intermediate result (optional)
            # intermediate_filename = os.path.join(outputs_folder, f'{job_id}_part_{current_sampling_step}.mp4')
            # save_bcthw_as_mp4(history_pixels, intermediate_filename, crf=mp4_crf, frame_rate=30)

        # Final save
        update_progress("Saving final video...", 98, 0, steps)
        final_filename = os.path.join(outputs_folder, f"{job_id}.mp4")
        save_bcthw_as_mp4(history_pixels, final_filename, crf=mp4_crf, fps=30)  # Use fps instead of frame_rate

        # Update job status and final progress
        update_progress("Finished", 100, steps, steps)  # Mark 100% progress
        queue_manager.update_job_status(job_id, "completed")
        print(f"Job {job_id} completed successfully. Output: {final_filename}")

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
        print(f"Worker finished for job {job_id}")


# Example usage (for testing purposes, would be called by a background task runner)
if __name__ == "__main__":
    print("Worker module loaded. Contains the 'worker' function.")
    # Add test code here if needed, e.g., creating a dummy job and models
    # and calling worker(dummy_job, dummy_models)
    pass
