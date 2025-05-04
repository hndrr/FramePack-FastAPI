import torch
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel

from diffusers_helper.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

# Determine VRAM mode
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60
print(f'Models: Free VRAM {free_mem_gb} GB')
print(f'Models: High-VRAM Mode: {high_vram}')

# Consider making model paths configurable via settings.py later
HUNYUAN_VIDEO_BASE = "hunyuanvideo-community/HunyuanVideo"
FLUX_REDUX_BASE = "lllyasviel/flux_redux_bfl"
FRAMEPACK_BASE = 'lllyasviel/FramePackI2V_HY'
FRAMEPACK_F1 = 'lllyasviel/FramePack_F1_I2V_HY_20250503'


def load_models():
    """
    Loads all necessary models and tokenizers (excluding LoRA).
    LoRA is loaded dynamically by the worker based on job parameters.
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
    transformer_base = HunyuanVideoTransformer3DModelPacked.from_pretrained(FRAMEPACK_BASE, torch_dtype=torch.bfloat16).cpu()
    print("Loading F1 transformer...")
    transformer_f1 = HunyuanVideoTransformer3DModelPacked.from_pretrained(FRAMEPACK_F1, torch_dtype=torch.bfloat16).cpu()

    # Set models to evaluation mode
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer_base.eval()
    transformer_f1.eval()

    # Apply VRAM optimizations if needed
    if not high_vram:
        print("Applying low VRAM optimizations (slicing/tiling for VAE)...")
        vae.enable_slicing()
        vae.enable_tiling()

    # Configure transformer output quality
    transformer_base.high_quality_fp32_output_for_inference = True
    transformer_f1.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True (for both models)')

    # Set model dtypes
    transformer_base.to(dtype=torch.bfloat16)
    transformer_f1.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    # Disable gradients
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer_base.requires_grad_(False)
    transformer_f1.requires_grad_(False)

    # LoRA loading moved to worker function

    # Move models to appropriate device based on VRAM
    if not high_vram:
        print("Installing DynamicSwap for low VRAM mode...")
        DynamicSwapInstaller.install_model(transformer_base, device=gpu)
        DynamicSwapInstaller.install_model(transformer_f1, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        # Note: VAE, text_encoder_2, image_encoder will be loaded/offloaded as needed by the worker in low VRAM mode
        print("DynamicSwap installed for transformers and text_encoder.")
    else:
        print("Moving all models to GPU for high VRAM mode...")
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer_base.to(gpu)
        transformer_f1.to(gpu)
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
        "transformer_base": transformer_base,
        "transformer_f1": transformer_f1,
        "high_vram": high_vram
    }


# Example usage (for testing purposes)
if __name__ == '__main__':
    # You might need to handle HF_HOME environment variable here if running standalone
    # os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '../hf_download')))

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--lora", type=str, default=None, help="Lora path for testing")
    # args = parser.parse_args()

    print("Testing model loading...")
    loaded_models = load_models()
    print(f"Models loaded: {list(loaded_models.keys())}")
    print(f"High VRAM mode detected: {loaded_models['high_vram']}")
    # Add more checks if needed
    if loaded_models['high_vram']:
        print(f"Base Transformer device: {loaded_models['transformer_base'].device}")
        print(f"F1 Transformer device: {loaded_models['transformer_f1'].device}")
    else:
        print("Running in low VRAM mode, models might be on CPU or dynamically swapped.")

    print("Model loading test finished.")


# Placeholder function for unloading models and cleaning up resources
def unload_models(models_dict):
    """
    Explicitly unloads models and releases resources, especially GPU memory.
    """
    print("Unloading models...")
    model_keys = ["vae", "text_encoder", "text_encoder_2", "image_encoder", "transformer_base", "transformer_f1"]
    for key in model_keys:
        if key in models_dict:
            try:
                del models_dict[key]
                print(f"Removed reference to model: {key}")
            except Exception as e:
                print(f"Error removing reference to model {key}: {e}")

    # Clear GPU cache if CUDA is available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            print("Cleared PyTorch CUDA cache.")
        except Exception as e:
            print(f"Error clearing CUDA cache: {e}")
    else:
        print("CUDA not available, skipping cache clear.")

    # Optionally remove other non-model items if needed
    other_keys = ["tokenizer", "tokenizer_2", "feature_extractor", "high_vram"]
    for key in other_keys:
        if key in models_dict:
            try:
                del models_dict[key]
                print(f"Removed reference to: {key}")
            except Exception as e:
                print(f"Error removing reference to {key}: {e}")

    print("Model unloading complete.")
