import os
import torch
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
import argparse  # Keep for now if LoRA path comes from args, consider moving to settings

from diffusers_helper.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller  # Removed cpu, load_model_as_complete
from diffusers_helper.load_lora import load_lora
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

# Determine VRAM mode
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60  # Threshold might need adjustment
print(f'Models: Free VRAM {free_mem_gb} GB')
print(f'Models: High-VRAM Mode: {high_vram}')

# Consider making model paths configurable via settings.py later
HUNYUAN_VIDEO_BASE = "hunyuanvideo-community/HunyuanVideo"
FLUX_REDUX_BASE = "lllyasviel/flux_redux_bfl"
FRAMEPACK_BASE = 'lllyasviel/FramePackI2V_HY'


def load_models(lora_path: str = None):
    """
    Loads all necessary models and tokenizers.
    Args:
        lora_path (str, optional): Path to the LoRA file. Defaults to None.
    Returns:
        dict: A dictionary containing the loaded models and tokenizers.
    """
    print("Loading models...")

    # Load Tokenizers and Text Encoders
    print("Loading tokenizers and text encoders...")
    tokenizer = LlamaTokenizerFast.from_pretrained(HUNYUAN_VIDEO_BASE, subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained(HUNYUAN_VIDEO_BASE, subfolder='tokenizer_2')
    text_encoder = LlamaModel.from_pretrained(HUNYUAN_VIDEO_BASE, subfolder='text_encoder', torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained(HUNYUAN_VIDEO_BASE, subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLHunyuanVideo.from_pretrained(HUNYUAN_VIDEO_BASE, subfolder='vae', torch_dtype=torch.float16).cpu()

    # Load Image Encoder and Feature Extractor
    print("Loading image encoder and feature extractor...")
    feature_extractor = SiglipImageProcessor.from_pretrained(FLUX_REDUX_BASE, subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained(FLUX_REDUX_BASE, subfolder='image_encoder', torch_dtype=torch.float16).cpu()

    # Load Transformer
    print("Loading transformer...")
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(FRAMEPACK_BASE, torch_dtype=torch.bfloat16).cpu()

    # Set models to evaluation mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    # Apply VRAM optimizations if needed
    if not high_vram:
        print("Applying low VRAM optimizations (slicing/tiling for VAE)...")
        vae.enable_slicing()
        vae.enable_tiling()

    # Configure transformer output quality
    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')

    # Set model dtypes
    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    # Disable gradients
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    # Load LoRA if specified
    if lora_path:
        if os.path.exists(lora_path):
            print(f"Loading LoRA from: {lora_path}")
            lora_dir, lora_name = os.path.split(lora_path)
            try:
                transformer = load_lora(transformer, lora_dir, lora_name)
                print("LoRA loaded successfully.")
            except Exception as e:
                print(f"Error loading LoRA: {e}")
                # Decide how to handle LoRA loading failure (e.g., continue without LoRA, raise error)
        else:
            print(f"Warning: LoRA path specified but not found: {lora_path}")

    # Move models to appropriate device based on VRAM
    if not high_vram:
        print("Installing DynamicSwap for low VRAM mode...")
        # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but potentially faster
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        # Note: VAE, text_encoder_2, image_encoder will be loaded/offloaded as needed by the worker in low VRAM mode
        print("DynamicSwap installed for transformer and text_encoder.")
    else:
        print("Moving all models to GPU for high VRAM mode...")
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)
        print("All models moved to GPU.")

    print("Model loading complete.")

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "feature_extractor": feature_extractor,
        "image_encoder": image_encoder,
        "transformer": transformer,
        "high_vram": high_vram  # Include vram mode info
    }


# Example usage (for testing purposes)
if __name__ == '__main__':
    # You might need to handle HF_HOME environment variable here if running standalone
    # os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '../hf_download')))

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", type=str, default=None, help="Lora path for testing")
    args = parser.parse_args()

    print("Testing model loading...")
    loaded_models = load_models(lora_path=args.lora)
    print(f"Models loaded: {list(loaded_models.keys())}")
    print(f"High VRAM mode detected: {loaded_models['high_vram']}")
    # Add more checks if needed, e.g., checking model devices
    if loaded_models['high_vram']:
        print(f"Transformer device: {loaded_models['transformer'].device}")
    else:
        print("Running in low VRAM mode, models might be on CPU or dynamically swapped.")

    print("Model loading test finished.")