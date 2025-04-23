import os
import json
import traceback
import uuid
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import gradio as gr
import einops
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.load_lora import load_lora
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

# Path to the quick prompts JSON file
PROMPT_FILE = os.path.join(os.getcwd(), 'quick_prompts.json')

# Queue file path
QUEUE_FILE = os.path.join(os.getcwd(), 'job_queue.json')

# Temp directory for queue images
temp_queue_images = os.path.join(os.getcwd(), 'temp_queue_images')
os.makedirs(temp_queue_images, exist_ok=True)

# Default prompts
DEFAULT_PROMPTS = [
    {'prompt': 'The girl dances gracefully, with clear movements, full of charm.', 'length': 5.0},
    {'prompt': 'A character doing some simple body movements.', 'length': 5.0},
]

# Load existing prompts or create the file with defaults
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, 'r') as f:
        quick_prompts = json.load(f)
else:
    quick_prompts = DEFAULT_PROMPTS.copy()
    with open(PROMPT_FILE, 'w') as f:
        json.dump(quick_prompts, f, indent=2)


@dataclass
class QueuedJob:
    prompt: str
    image_path: str
    video_length: float
    job_id: str  # Changed to string for hex ID
    seed: int
    use_teacache: bool
    gpu_memory_preservation: float
    steps: int
    cfg: float
    gs: float
    rs: float
    status: str = "pending"
    thumbnail: str = ""
    mp4_crf: float = 16

    def to_dict(self):
        try:
            return {
                'prompt': self.prompt,
                'image_path': self.image_path,
                'video_length': self.video_length,
                'job_id': self.job_id,
                'seed': self.seed,
                'use_teacache': self.use_teacache,
                'gpu_memory_preservation': self.gpu_memory_preservation,
                'steps': self.steps,
                'cfg': self.cfg,
                'gs': self.gs,
                'rs': self.rs,
                'status': self.status,
                'thumbnail': self.thumbnail,
                'mp4_crf': self.mp4_crf
            }
        except Exception as e:
            print(f"Error converting job to dict: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                prompt=data['prompt'],
                image_path=data['image_path'],
                video_length=data['video_length'],
                job_id=data['job_id'],
                seed=data['seed'],
                use_teacache=data['use_teacache'],
                gpu_memory_preservation=data['gpu_memory_preservation'],
                steps=data['steps'],
                cfg=data['cfg'],
                gs=data['gs'],
                rs=data['rs'],
                status=data['status'],
                thumbnail=data['thumbnail'],
                mp4_crf=data['mp4_crf']
            )
        except Exception as e:
            print(f"Error creating job from dict: {str(e)}")
            return None


# Initialize job queue as a list
job_queue = []


def save_queue():
    try:
        jobs = []
        for job in job_queue:
            job_dict = job.to_dict()
            if job_dict is not None:
                jobs.append(job_dict)

        file_path = os.path.abspath(QUEUE_FILE)
        with open(file_path, 'w') as f:
            json.dump(jobs, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving queue: {str(e)}")
        traceback.print_exc()
        return False


def load_queue():
    try:
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'r') as f:
                jobs = json.load(f)
            # Clear existing queue and load jobs from file
            job_queue.clear()
            for job_data in jobs:
                job = QueuedJob.from_dict(job_data)
                if job is not None:
                    job_queue.append(job)
            return job_queue
        return []
    except Exception as e:
        print(f"Error loading queue: {str(e)}")
        traceback.print_exc()
        return []


# Load existing queue on startup
job_queue = load_queue()


def save_image_to_temp(image: np.ndarray, job_id: str) -> str:
    """Save image to temp directory and return the path"""
    try:
        # Convert numpy array to PIL Image
        # Remove single-dimensional entries from the shape of an array
        squeezed_image = np.squeeze(image)
        pil_image = Image.fromarray(squeezed_image)
        # Create unique filename using hex ID
        filename = f"queue_image_{job_id}.png"
        filepath = os.path.join(temp_queue_images, filename)
        # Save image
        pil_image.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return ""


def add_to_queue(prompt, image, video_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, status="pending", mp4_crf=16):
    try:
        # Generate a unique hex ID for the job
        job_id = uuid.uuid4().hex[:8]
        # Save image to temp directory and get path
        image_array = np.array(image)
        image_path = save_image_to_temp(image_array, job_id)
        if not image_path:
            return None

        job = QueuedJob(
            prompt=prompt,
            image_path=image_path,
            video_length=video_length,
            job_id=job_id,
            seed=seed,
            use_teacache=use_teacache,
            gpu_memory_preservation=gpu_memory_preservation,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            status=status,
            mp4_crf=mp4_crf
        )
        job_queue.append(job)
        save_queue()  # Save immediately after adding
        return job_id
    except Exception as e:
        print(f"Error adding job to queue: {str(e)}")
        traceback.print_exc()
        return None


def get_next_job():
    try:
        if job_queue:
            job = job_queue.pop(0)  # Remove and return first job
            save_queue()  # Save after removing job
            return job
        return None
    except Exception as e:
        print(f"Error getting next job: {str(e)}")
        traceback.print_exc()
        return None


def update_queue_display():
    try:
        queue_data = []
        for job in job_queue:
            # Create thumbnail if it doesn't exist
            if not job.thumbnail and job.image_path:
                try:
                    # Load and resize image for thumbnail
                    img = Image.open(job.image_path)
                    width, height = img.size
                    new_height = 200
                    new_width = int((new_height / height) * width)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    thumb_path = os.path.join(temp_queue_images, f"thumb_{job.job_id}.png")
                    img.save(thumb_path)
                    job.thumbnail = thumb_path
                    save_queue()
                except Exception as e:
                    print(f"Error creating thumbnail: {str(e)}")
                    job.thumbnail = ""

            # Add job data to display
            if job.thumbnail:
                caption = f"{job.status}\n\nPrompt: {job.prompt}...\n\nLength: {job.video_length}s"
                queue_data.append((job.thumbnail, caption))

        return queue_data
    except Exception as e:
        print(f"Error updating queue display: {str(e)}")
        traceback.print_exc()
        return []


# Quick prompts management functions


def get_default_prompt():
    try:
        if quick_prompts and len(quick_prompts) > 0:
            return quick_prompts[0]['prompt'], quick_prompts[0]['length']
        return "", 5.0
    except Exception as e:
        print(f"Error getting default prompt: {str(e)}")
        return "", 5.0


def save_quick_prompt(prompt_text, video_length):
    global quick_prompts
    if prompt_text:
        # Check if prompt already exists
        for item in quick_prompts:
            if item['prompt'] == prompt_text:
                item['length'] = video_length
                break
        else:
            quick_prompts.append({'prompt': prompt_text, 'length': video_length})

        with open(PROMPT_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)
    # Keep the text in the prompt box and set it as selected in quick list
    return prompt_text, gr.update(choices=[item['prompt'] for item in quick_prompts], value=prompt_text), video_length


def delete_quick_prompt(prompt_text):
    global quick_prompts
    if prompt_text:
        quick_prompts = [item for item in quick_prompts if item['prompt'] != prompt_text]
        with open(PROMPT_FILE, 'w') as f:
            json.dump(quick_prompts, f, indent=2)
    # Clear the prompt box and quick list selection
    return "", gr.update(choices=[item['prompt'] for item in quick_prompts], value=None), 5.0


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lora", type=str, default=None, help="Lora path")
parser.add_argument("--lora_scale", type=float, default=1.0, help="Lora scale factor (weight)")
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if args.lora:
    lora = args.lora
    lora_path, lora_name = os.path.split(lora)
    print("Loading lora")
    transformer = load_lora(transformer, lora_path, lora_name, lora_scale=args.lora_scale)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    # Ensure input_image is a NumPy array
    if isinstance(input_image, list):
        input_image = np.array(input_image)

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...'))))

        # Ensure the image shape is 3D (H, W, C) by squeezing extra dimensions
        input_image = np.squeeze(input_image)
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))

        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'), pnginfo=metadata)
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

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

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
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

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                break
    except Exception as e:
        print(f"Error during worker execution: {e}")
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def create_status_thumbnail(image_path, status, border_color, status_text):
    """Create a thumbnail with status-specific border and text"""
    try:
        # Load and resize image for thumbnail
        img = Image.open(image_path)
        width, height = img.size
        new_height = 200
        new_width = int((new_height / height) * width)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Add border
        border_size = 5
        img_with_border = Image.new('RGB',
                                    (img.width + border_size * 2, img.height + border_size * 2),
                                    border_color)
        img_with_border.paste(img, (border_size, border_size))

        # Add status text
        draw = ImageDraw.Draw(img_with_border)
        # Use smaller font size for RUNNING text
        font_size = 30 if status_text == "RUNNING" else 40
        font = ImageFont.truetype("arial.ttf", font_size)  # You might need to adjust font path
        text = status_text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position text in center
        x = (img_with_border.width - text_width) // 2
        y = (img_with_border.height - text_height) // 2

        # Draw text with black outline
        for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + offset[0], y + offset[1]), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        return img_with_border
    except Exception as e:
        print(f"Error creating status thumbnail: {str(e)}")
        traceback.print_exc()
        return None


def mark_job_processing(job):
    """Mark a job as processing and update its thumbnail with a red border and RUNNING text"""
    try:
        job.status = "processing"

        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)

        # Create new thumbnail with processing status
        if job.image_path and os.path.exists(job.image_path):
            # Create thumbnail path if it doesn't exist
            if not job.thumbnail:
                job.thumbnail = os.path.join(temp_queue_images, f"thumb_{job.job_id}.png")

            new_thumbnail = create_status_thumbnail(
                job.image_path,
                "processing",
                (255, 0, 0),  # Red color
                "RUNNING"
            )
            if new_thumbnail:
                new_thumbnail.save(job.thumbnail)

    except Exception as e:
        print(f"Error modifying thumbnail: {str(e)}")
        traceback.print_exc()


def mark_job_completed(job):
    """Mark a job as completed and update its thumbnail with a green border and DONE text"""
    try:
        job.status = "completed"

        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)

        # Create new thumbnail with completed status
        if job.image_path and os.path.exists(job.image_path):
            # Create thumbnail path if it doesn't exist
            if not job.thumbnail:
                job.thumbnail = os.path.join(temp_queue_images, f"thumb_{job.job_id}.png")

            new_thumbnail = create_status_thumbnail(
                job.image_path,
                "completed",
                (0, 255, 0),  # Green color
                "DONE"
            )
            if new_thumbnail:
                new_thumbnail.save(job.thumbnail)

    except Exception as e:
        print(f"Error modifying thumbnail: {str(e)}")
        traceback.print_exc()


def mark_job_pending(job):
    """Mark a job as pending and update its thumbnail to a clean version without border or text"""
    try:
        job.status = "pending"

        # Delete existing thumbnail if it exists
        if job.thumbnail and os.path.exists(job.thumbnail):
            os.remove(job.thumbnail)

        # Create new clean thumbnail
        if job.image_path and os.path.exists(job.image_path):
            # Create thumbnail path if it doesn't exist
            if not job.thumbnail:
                job.thumbnail = os.path.join(temp_queue_images, f"thumb_{job.job_id}.png")

            # Load and resize image for thumbnail
            img = Image.open(job.image_path)
            width, height = img.size
            new_height = 200
            new_width = int((new_height / height) * width)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save clean thumbnail
            img.save(job.thumbnail)

    except Exception as e:
        print(f"Error modifying thumbnail: {str(e)}")
        traceback.print_exc()


def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf):
    global stream

    # Initialize variables
    output_filename = None
    job_id = None

    # Convert Gallery tuples to numpy arrays if needed
    if input_image is None:
        input_image = None
    elif isinstance(input_image, list):
        input_image = [np.array(Image.open(img[0])) for img in input_image]
    else:
        # Single image case
        input_image = np.array(Image.open(input_image[0]))

    # Handle multiple input images
    if isinstance(input_image, list) and len(input_image) > 1:
        # For multiple images, add each as a separate job to the queue
        for i, img in enumerate(input_image):
            status = "just_added" if i == 0 else "pending"  # First image gets just_added, rest get pending
            job_id = add_to_queue(
                prompt=prompt,
                image=img,
                video_length=total_second_length,
                seed=seed,
                use_teacache=use_teacache,
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status=status,
                mp4_crf=mp4_crf
            )

        # After adding all jobs, process the first one
        input_image = input_image[0]

    # Determine which job to process
    if input_image is not None:
        # Check for just_added jobs first
        just_added_jobs = [job for job in job_queue if job.status == "just_added"]
        if just_added_jobs:
            next_job = just_added_jobs[0]
            mark_job_processing(next_job)  # Use new function to mark as processing
            save_queue()
            job_id = next_job.job_id

            try:
                process_image = np.array(Image.open(next_job.image_path))
            except Exception as e:
                print(f"ERROR loading image: {str(e)}")
                traceback.print_exc()
                raise

            process_prompt = next_job.prompt
            process_seed = next_job.seed
            process_length = next_job.video_length
            process_steps = next_job.steps
            process_cfg = next_job.cfg
            process_gs = next_job.gs
            process_rs = next_job.rs
            process_preservation = next_job.gpu_memory_preservation
            process_teacache = next_job.use_teacache
        else:
            # Process input image
            job_id = add_to_queue(
                prompt=prompt,
                image=input_image,
                video_length=total_second_length,
                seed=seed,
                use_teacache=use_teacache,
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status="processing"
            )
            # Find and mark the new job as processing
            for job in job_queue:
                if job.job_id == job_id:
                    mark_job_processing(job)
                    break
            process_image = input_image
            process_prompt = prompt
            process_seed = seed
            process_length = total_second_length
            process_steps = steps
            process_cfg = cfg
            process_gs = gs
            process_rs = rs
            process_preservation = gpu_memory_preservation
            process_teacache = use_teacache
    else:
        # Check for pending jobs
        pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
        if not pending_jobs:
            assert input_image is not None, 'No input image!'

        # Process first pending job
        next_job = pending_jobs[0]
        mark_job_processing(next_job)  # Use new function to mark as processing
        save_queue()
        job_id = next_job.job_id

        try:
            process_image = np.array(Image.open(next_job.image_path))
        except Exception as e:
            print(f"ERROR loading image: {str(e)}")
            traceback.print_exc()
            raise

        process_prompt = next_job.prompt
        process_seed = next_job.seed
        process_length = next_job.video_length
        process_steps = next_job.steps
        process_cfg = next_job.cfg
        process_gs = next_job.gs
        process_rs = next_job.rs
        process_preservation = next_job.gpu_memory_preservation
        process_teacache = next_job.use_teacache

    # Start processing
    stream = AsyncStream()
    async_run(worker, process_image, process_prompt, n_prompt, process_seed,
              process_length, latent_window_size, process_steps,
              process_cfg, process_gs, process_rs,
              process_preservation, process_teacache, mp4_crf)

    # Initial yield with updated queue display and button states
    yield (
        None,  # result_video
        None,  # preview_image
        '',    # progress_desc
        '',    # progress_bar
        gr.update(interactive=False),  # start_button
        gr.update(interactive=True),   # end_button
        gr.update(interactive=True),   # queue_button (always enabled)
        update_queue_display()         # queue_display
    )

    # Process output queue
    while True:
        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield (
                output_filename,  # result_video
                gr.update(),  # preview_image
                gr.update(),  # progress_desc
                gr.update(),  # progress_bar
                gr.update(interactive=False),  # start_button
                gr.update(interactive=True),   # end_button
                gr.update(interactive=True),   # queue_button (always enabled)
                update_queue_display()         # queue_display
            )

        if flag == 'progress':
            preview, desc, html = data
            yield (
                gr.update(),  # result_video
                gr.update(visible=True, value=preview),  # preview_image
                desc,  # progress_desc
                html,  # progress_bar
                gr.update(interactive=False),  # start_button
                gr.update(interactive=True),   # end_button
                gr.update(interactive=True),   # queue_button (always enabled)
                update_queue_display()         # queue_display
            )

        if flag == 'end':
            # Find and mark all processing jobs as completed
            for job in job_queue:
                if job.status == "processing":
                    mark_job_completed(job)
                    save_queue()
                    break

            # Then check if we should continue processing (only if end button wasn't clicked)
            if not stream.input_queue.top() == 'end':
                # Find next job to process
                next_job = None

                # First check for pending jobs
                pending_jobs = [job for job in job_queue if job.status.lower() == "pending"]
                if pending_jobs:
                    next_job = pending_jobs[0]
                else:
                    # If no pending jobs, check for just_added jobs
                    just_added_jobs = [job for job in job_queue if job.status == "just_added"]
                    if just_added_jobs:
                        next_job = just_added_jobs[0]

                if next_job:
                    # Update next job status to processing
                    mark_job_processing(next_job)  # Use new function to mark as processing
                    save_queue()

                    try:
                        next_image = np.array(Image.open(next_job.image_path))
                    except Exception as e:
                        print(f"ERROR loading next image: {str(e)}")
                        traceback.print_exc()
                        raise

                    # Process next job
                    async_run(worker, next_image, next_job.prompt, n_prompt, next_job.seed,
                              next_job.video_length, latent_window_size, next_job.steps,
                              next_job.cfg, next_job.gs, next_job.rs,
                              next_job.gpu_memory_preservation, next_job.use_teacache, mp4_crf)
                else:
                    job_queue[:] = [job for job in job_queue if job.status != "completed"]
                    save_queue()
                    # No more jobs, return to initial state
                    yield (
                        output_filename,  # result_video
                        gr.update(visible=False),  # preview_image
                        gr.update(),  # progress_desc
                        '',  # progress_bar
                        gr.update(interactive=True),  # start_button
                        gr.update(interactive=False),  # end_button
                        gr.update(interactive=True),  # queue_button
                        update_queue_display()  # queue_display
                    )
                    break
            else:
                # End button was clicked, stop processing
                job_queue[:] = [job for job in job_queue if job.status != "completed"]
                save_queue()
                yield (
                    output_filename,  # result_video
                    gr.update(visible=False),  # preview_image
                    gr.update(),  # progress_desc
                    '',  # progress_bar
                    gr.update(interactive=True),  # start_button
                    gr.update(interactive=False),  # end_button
                    gr.update(interactive=True),  # queue_button
                    update_queue_display()  # queue_display
                )
                break


def end_process():
    """Handle end generation button click - stop all processes and change all processing jobs to pending jobs"""
    try:
        # First send the end signal to stop all processes
        stream.input_queue.push('end')

        # Find and update all processing jobs
        jobs_changed = 0
        processing_job = None
        job_queue[:] = [job for job in job_queue if job.status != "completed"]

        # First find the processing job
        for job in job_queue:
            if job.status == "processing":
                processing_job = job
                break

        # Then process all jobs
        for job in job_queue:
            if job.status == "processing":
                mark_job_pending(job)  # Use new function to mark as pending
                jobs_changed += 1

        # If we found a processing job, move it to the top
        if processing_job:
            job_queue.remove(processing_job)
            job_queue.insert(0, processing_job)

        save_queue()
        return (
            update_queue_display(),  # queue_display
            gr.update(interactive=True)  # queue_button (always enabled)
        )
    except Exception as e:
        print(f"Error in end_process: {str(e)}")
        traceback.print_exc()
        return [], gr.update(interactive=True)  # queue_button (always enabled)


def add_to_queue_handler(input_image, prompt, total_second_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf):
    """Handle adding a new job to the queue"""
    if input_image is None or not prompt:
        return [], gr.update(interactive=True)  # queue_button (always enabled)

    try:
        # Convert Gallery tuples to numpy arrays if needed
        if isinstance(input_image, list):
            # Multiple images case
            input_images = [np.array(Image.open(img[0])) for img in input_image]

            # Change any existing just_added jobs to pending
            for job in job_queue:
                if job.status == "just_added":
                    job.status = "pending"

            # Add each image as a separate job with pending status
            for img in input_images:
                job_id = add_to_queue(
                    prompt=prompt,
                    image=img,
                    video_length=total_second_length,
                    seed=seed,
                    use_teacache=use_teacache,
                    gpu_memory_preservation=gpu_memory_preservation,
                    steps=steps,
                    cfg=cfg,
                    gs=gs,
                    rs=rs,
                    status="pending",  # All images get pending status when using Add to Queue
                    mp4_crf=mp4_crf
                )
        else:
            # Single image case
            input_image = np.array(Image.open(input_image[0]))

            # Change any existing just_added jobs to pending
            for job in job_queue:
                if job.status == "just_added":
                    job.status = "pending"

            # Add single image job
            job_id = add_to_queue(
                prompt=prompt,
                image=input_image,
                video_length=total_second_length,
                seed=seed,
                use_teacache=use_teacache,
                gpu_memory_preservation=gpu_memory_preservation,
                steps=steps,
                cfg=cfg,
                gs=gs,
                rs=rs,
                status="pending",  # Single image gets pending status when using Add to Queue
                mp4_crf=mp4_crf
            )

        if job_id is not None:
            save_queue()  # Save after changing statuses
            return update_queue_display(), gr.update(interactive=True)  # queue_button (always enabled)
        else:
            return [], gr.update(interactive=True)  # queue_button (always enabled)
    except Exception as e:
        print(f"Error in add_to_queue_handler: {str(e)}")
        traceback.print_exc()
        return [], gr.update(interactive=True)  # queue_button (always enabled)


def cleanup_orphaned_files():
    """Clean up any temp files that don't correspond to jobs in the queue"""
    try:
        # Get all job files from queue
        job_files = set()
        for job in job_queue:
            if job.image_path:
                job_files.add(job.image_path)
            if job.thumbnail:
                job_files.add(job.thumbnail)

        # Get all files in temp directory
        temp_files = set()
        for root, _, files in os.walk(temp_queue_images):
            for file in files:
                temp_files.add(os.path.join(root, file))

        # Find orphaned files (in temp but not in queue)
        orphaned_files = temp_files - job_files

        # Delete orphaned files
        for file in orphaned_files:
            try:
                os.remove(file)
                print(f"Deleted orphaned file: {file}")
            except Exception as e:
                print(f"Error deleting file {file}: {str(e)}")
    except Exception as e:
        print(f"Error in cleanup_orphaned_files: {str(e)}")
        traceback.print_exc()


def reset_processing_jobs():
    """Reset any processing or just_added jobs to pending and move them to top of queue"""
    job_queue[:] = [job for job in job_queue if job.status != "completed"]
    try:
        print("\n=== DEBUG: Reset Processing Jobs Called ===")
        # Find all processing and just_added jobs
        jobs_to_move = []
        for job in job_queue:
            if job.status in ("processing", "just_added"):
                print(f"Found job {job.job_id} with status {job.status}")
                jobs_to_move.append(job)

        # Remove these jobs from their current positions
        for job in jobs_to_move:
            job_queue.remove(job)
            mark_job_pending(job)  # Use new function to mark as pending
            print(f"Changed job {job.job_id} status to pending")

        # Add them back at the top in reverse order (so they maintain their relative order)
        for job in reversed(jobs_to_move):
            job_queue.insert(0, job)
            print(f"Moved job {job.job_id} to top of queue")

        save_queue()
        print(f"Queue saved with {len(jobs_to_move)} jobs moved to top")
    except Exception as e:
        print(f"Error in reset_processing_jobs: {str(e)}")
        traceback.print_exc()


def delete_job(job_id):
    """Delete a job from the queue and its associated files"""
    try:
        # Find and remove job from queue
        for job in job_queue:
            if job.job_id == job_id:
                # Delete associated files
                if os.path.exists(job.image_path):
                    os.remove(job.image_path)
                if os.path.exists(job.thumbnail):
                    os.remove(job.thumbnail)
                job_queue.remove(job)
                break
        save_queue()
        return update_queue_display()
    except Exception as e:
        print(f"Error deleting job: {str(e)}")
        traceback.print_exc()
        return update_queue_display()


# Add these calls at startup
reset_processing_jobs()
cleanup_orphaned_files()
# Add custom CSS for the queue display
css = make_progress_bar_css() + """
.gradio-gallery-container {
    max-height: 600px !important;
    overflow-y: auto !important;
    padding: 10px;
}
.gradio-gallery-container::-webkit-scrollbar {
    width: 8px !important;
}
.gradio-gallery-container::-webkit-scrollbar-track {
    background: #f0f0f0 !important;
}
.gradio-gallery-container::-webkit-scrollbar-thumb {
    background-color: #666 !important;
    border-radius: 4px !important;
}
.queue-gallery .gallery-item {
    margin: 5px;
}
"""
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack')
    with gr.Row():
        with gr.Column():
            input_image = gr.Gallery(
                label="Image",
                height=320,
                columns=3,
                object_fit="contain"
            )
            prompt = gr.Textbox(label="Prompt", value='')
            save_prompt_button = gr.Button("Save Prompt")
            delete_prompt_button = gr.Button("Delete Selected Prompt")
            quick_list = gr.Dropdown(
                label="Quick List",
                choices=[item['prompt'] for item in quick_prompts],
                value=quick_prompts[0]['prompt'] if quick_prompts else None,
                allow_custom_value=True
            )

            with gr.Group():
                use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                mp4_crf = gr.Slider(label="MP4 Compression", minimum=0, maximum=100, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Change to 16 if you get black outputs. ")

            # Set default prompt and length
            default_prompt, default_length = get_default_prompt()
            prompt.value = default_prompt
            total_second_length.value = default_length

            save_prompt_button.click(
                save_quick_prompt,
                inputs=[prompt, total_second_length],
                outputs=[prompt, quick_list, total_second_length],
                queue=False
            )
            delete_prompt_button.click(
                delete_quick_prompt,
                inputs=[quick_list],
                outputs=[prompt, quick_list, total_second_length],
                queue=False
            )
            quick_list.change(
                lambda x: (x, next((item['length'] for item in quick_prompts if item['prompt'] == x), 5.0)) if x else ("", 5.0),
                inputs=[quick_list],
                outputs=[prompt, total_second_length],
                queue=False
            )

            # Add JavaScript to set default prompt on page load
            block.load(
                fn=lambda: (default_prompt, default_length),
                inputs=None,
                outputs=[prompt, total_second_length],
                queue=False
            )

        with gr.Column():
            with gr.Row():
                start_button = gr.Button(value="Start Generation", interactive=True)
                end_button = gr.Button(value="End Generation", interactive=False)
                queue_button = gr.Button(value="Add to Queue", interactive=True)

            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
            gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')

            queue_display = gr.Gallery(
                label="Job Queue",
                show_label=True,
                columns=3,
                object_fit="contain",
                elem_classes=["queue-gallery"],
                allow_preview=True,
                show_download_button=False,
                container=True
            )

            # Load queue on startup
            block.load(
                fn=update_queue_display,
                inputs=None,
                outputs=[queue_display],
                queue=False
            )

    gr.HTML('<div style="text-align:center; margin-top:20px;">Share your results and find ideas at the <a href="https://x.com/search?q=framepack&f=live" target="_blank">FramePack Twitter (X) thread</a></div>')

    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf]
    start_button.click(
        fn=process,
        inputs=ips,
        outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, queue_button, queue_display]
    )
    end_button.click(
        fn=end_process,
        outputs=[queue_display, queue_button]
    )
    queue_button.click(
        fn=add_to_queue_handler,
        inputs=[input_image, prompt, total_second_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, mp4_crf],
        outputs=[queue_display, queue_button]
    )

block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
