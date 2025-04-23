import os
import torch
import numpy as np
# import einops  # Unused
# import safetensors.torch as sf  # Unused
from PIL import Image  # Removed ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import traceback

# Assuming models and tokenizers are loaded elsewhere and passed or accessed globally/via context
# This will be refined when creating models.py and integrating
# from diffusers import AutoencoderKLHunyuanVideo
# from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode  # vae_decode_fake might be needed too
# from diffusers_helper.load_lora import load_lora  # Unused
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, resize_and_center_crop, soft_append_bcthw  # Removed generate_timestamp, Added soft_append_bcthw
# from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked  # Unused
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, fake_diffusers_current_device, unload_complete_models, load_model_as_complete  # Removed cpu, offload_..., DynamicSwapInstaller
# from diffusers_helper.thread_utils import AsyncStream, async_run  # Removed stream
# from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html  # Unused
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

# Assuming queue_manager is available for status updates (relative import)
from . import queue_manager
from . import settings  # Import settings to get OUTPUTS_DIR

# Define output folder using settings
outputs_folder = settings.OUTPUTS_DIR
# os.makedirs(outputs_folder, exist_ok=True) # Directory creation handled in settings.py

# Determine VRAM mode - consider moving to settings.py or detecting dynamically
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60  # Threshold might need adjustment
print(f'Worker: Free VRAM {free_mem_gb} GB')
print(f'Worker: High-VRAM Mode: {high_vram}')


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
    # latent_window_size = job.latent_window_size  # Assuming this might be added or defaulted
    latent_window_size = 16  # Default value from demo_gradio
    steps = job.steps
    cfg = job.cfg
    gs = job.gs
    rs = job.rs
    gpu_memory_preservation = job.gpu_memory_preservation
    use_teacache = job.use_teacache
    mp4_crf = job.mp4_crf
    job_id = job.job_id

    # Update job status to processing
    queue_manager.update_job_status(job_id, "processing")

    # Load models from the dictionary
    vae = models['vae']
    text_encoder = models['text_encoder']
    text_encoder_2 = models['text_encoder_2']
    image_encoder = models['image_encoder']
    transformer = models['transformer']
    tokenizer = models['tokenizer']
    tokenizer_2 = models['tokenizer_2']
    feature_extractor = models['feature_extractor']

    # Placeholder for progress update function (replaces stream.output_queue.push)
    def update_progress(step_info: str, percentage: float = 0.0):
        print(f"Job {job_id} Progress: {step_info} ({percentage:.1f}%)")
        # Here you could potentially update the job status with more detail
        # queue_manager.update_job_status(job_id, f"processing - {step_info}", ...)
        pass  # Replace with actual progress reporting if needed

    update_progress('Starting ...', 0)

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
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        # Clean GPU if not high_vram
        if not high_vram:
            update_progress('Cleaning GPU memory...', 0)
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        update_progress('Text encoding ...', 5)
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        update_progress('Image processing ...', 10)
        input_image = np.squeeze(input_image)  # Ensure 3D
        if input_image.ndim != 3 or input_image.shape[2] != 3:
            print(f"Error: Invalid image shape {input_image.shape} for job {job_id}")
            queue_manager.update_job_status(job_id, "failed - invalid image shape")
            return

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)  # Assuming default resolution
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # Save a copy of the processed input image (optional, but good for reference)
        processed_input_image_path = os.path.join(outputs_folder, f'{job_id}_input.png')
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        Image.fromarray(input_image_np).save(processed_input_image_path, pnginfo=metadata)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        update_progress('VAE encoding ...', 15)
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision encoding
        update_progress('CLIP Vision encoding ...', 20)
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype conversion
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        update_progress('Start sampling ...', 25)
        rnd = torch.Generator("cpu").manual_seed(seed)
        # num_frames = latent_window_size * 4 - 3 # Unused variable

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        else:
            latent_paddings = list(latent_paddings)  # Convert range_iterator to list

        sampling_step_count = len(latent_paddings)
        current_sampling_step = 0

        for latent_padding in latent_paddings:
            current_sampling_step += 1
            section_progress_start = 25 + (current_sampling_step - 1) * (70 / sampling_step_count)
            section_progress_end = 25 + current_sampling_step * (70 / sampling_step_count)

            # Check for cancellation signal (e.g., if job status is set to 'cancelled')
            current_job_status = queue_manager.get_job_by_id(job_id)
            if current_job_status and current_job_status.status == 'cancelled':
                print(f"Job {job_id} cancelled during sampling.")
                # No need to update status again, it's already 'cancelled'
                return

            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(f'Job {job_id}: latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')
            update_progress(f'Sampling section {current_sampling_step}/{sampling_step_count}', section_progress_start)

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            # K-Diffusion Sampling Callback
            def callback(d):
                step = d['i']
                total = d['total']
                current_progress = section_progress_start + (step / total) * (section_progress_end - section_progress_start) * 0.9  # Allocate 90% of section time to sampling
                update_progress(f'Sampling section {current_sampling_step}/{sampling_step_count} - Step {step+1}/{total}', current_progress)

                # Check for cancellation signal within callback
                current_job_status_inner = queue_manager.get_job_by_id(job_id)
                if current_job_status_inner and current_job_status_inner.status == 'cancelled':
                    print(f"Job {job_id} cancelled during sampling step {step+1}.")
                    raise InterruptedError("Job cancelled")  # Raise exception to stop sampling

                # Original callback logic (if any) can go here
                # Example: preview generation (might be complex for API)
                # if step % 10 == 0:
                #     latents = d['denoised']
                #     pixels = vae_decode_fake(latents, vae)  # Use fake decode for speed
                #     # Send preview update (how depends on API design)

            try:
                generated_latents = sample_hunyuan(
                    model=transformer,
                    seed=rnd.randint(0, 1 << 30),
                    steps=steps,
                    cfg=cfg,
                    gs=gs,
                    rs=rs,
                    sampler_name='dpmpp_2m_sde',  # Or make configurable
                    scheduler_name='karras',     # Or make configurable
                    latent_image=start_latent,
                    positive_llama=llama_vec,
                    negative_llama=llama_vec_n,
                    positive_clip_l_pooler=clip_l_pooler,
                    negative_clip_l_pooler=clip_l_pooler_n,
                    positive_image_encoder_hidden_states=image_encoder_last_hidden_state,
                    negative_image_encoder_hidden_states=torch.zeros_like(image_encoder_last_hidden_state),
                    llama_attention_mask=llama_attention_mask,
                    llama_attention_mask_negative=llama_attention_mask_n,
                    latent_padding_size=latent_padding_size,
                    latent_window_size=latent_window_size,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                    disable_pbar=True  # Disable internal progress bar
                )
            except InterruptedError:
                # Job was cancelled during sampling via callback
                return  # Exit worker function

            # Update history
            history_latents = generated_latents[:, :, - (1 + 2 + 16):, :, :].clone()

            # Decode and append frames
            update_progress(f'VAE decoding section {current_sampling_step}/{sampling_step_count}', section_progress_end - 1)  # Just before finishing section
            if not high_vram:
                unload_complete_models()
                load_model_as_complete(vae, target_device=gpu)

            pixels = vae_decode(generated_latents[:, :, latent_padding_size:latent_padding_size + latent_window_size], vae)

            if history_pixels is None:
                history_pixels = pixels
            else:
                history_pixels = soft_append_bcthw(history_pixels, pixels, soft_length=4)

            total_generated_latent_frames += latent_window_size

            # Save intermediate result (optional)
            # intermediate_filename = os.path.join(outputs_folder, f'{job_id}_part_{current_sampling_step}.mp4')
            # save_bcthw_as_mp4(history_pixels, intermediate_filename, crf=mp4_crf, frame_rate=30)

        # Final save
        update_progress('Saving final video...', 98)
        final_filename = os.path.join(outputs_folder, f'{job_id}.mp4')
        save_bcthw_as_mp4(history_pixels, final_filename, crf=mp4_crf, frame_rate=30)

        # Update job status to completed
        queue_manager.update_job_status(job_id, "completed")
        update_progress('Finished', 100)
        print(f"Job {job_id} completed successfully. Output: {final_filename}")

    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")
        traceback.print_exc()
        queue_manager.update_job_status(job_id, f"failed - {type(e).__name__}")
        update_progress(f'Failed: {type(e).__name__}', 100)

    finally:
        # Final GPU cleanup (optional, depends on worker lifecycle)
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
        print(f"Worker finished for job {job_id}")


# Example usage (for testing purposes, would be called by a background task runner)
if __name__ == '__main__':
    print("Worker module loaded. Contains the 'worker' function.")
    # Add test code here if needed, e.g., creating a dummy job and models
    # and calling worker(dummy_job, dummy_models)
    pass