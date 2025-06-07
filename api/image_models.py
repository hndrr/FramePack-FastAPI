"""
Image generation specific models and configurations
"""
from typing import Dict, Any, Tuple
import torch

# Image generation configuration
IMAGE_GENERATION_CONFIG = {
    "default_width": 1216,
    "default_height": 704,
    "max_batch_size": 8,
    "rope_batch_size": 4,
    "latent_channels": 16,
    "vae_scale_factor": 8,
    "time_scale_factor": 4,
    "transformer_embed_dim": 3072,
}


def get_rope_config(batch_size: int, height: int, width: int) -> Dict[str, Any]:
    """
    Get RoPE (Rotary Position Embedding) configuration for image generation
    
    Args:
        batch_size: Number of images in batch
        height: Image height in pixels
        width: Image width in pixels
    
    Returns:
        Dictionary with RoPE configuration
    """
    latent_height = height // IMAGE_GENERATION_CONFIG["vae_scale_factor"]
    latent_width = width // IMAGE_GENERATION_CONFIG["vae_scale_factor"]
    
    return {
        "batch_size": batch_size,
        "height": latent_height,
        "width": latent_width,
        "max_seq_length": latent_height * latent_width,
        "embed_dim": IMAGE_GENERATION_CONFIG["transformer_embed_dim"],
        "channels": IMAGE_GENERATION_CONFIG["latent_channels"],
    }


def get_image_latent_shape(
    batch_size: int, 
    height: int, 
    width: int,
    num_frames: int = 1
) -> Tuple[int, int, int, int, int]:
    """
    Get the shape of latents for image generation
    
    Args:
        batch_size: Number of images
        height: Image height in pixels
        width: Image width in pixels
        num_frames: Number of frames (1 for static image)
    
    Returns:
        Tuple of (batch_size, channels, frames, latent_height, latent_width)
    """
    vae_scale = IMAGE_GENERATION_CONFIG["vae_scale_factor"]
    channels = IMAGE_GENERATION_CONFIG["latent_channels"]
    
    return (
        batch_size,
        channels,
        num_frames,
        height // vae_scale,
        width // vae_scale
    )


def validate_image_dimensions(width: int, height: int) -> Tuple[int, int]:
    """
    Validate and adjust image dimensions to be compatible with VAE
    
    Args:
        width: Requested width
        height: Requested height
    
    Returns:
        Tuple of (adjusted_width, adjusted_height)
    """
    vae_scale = IMAGE_GENERATION_CONFIG["vae_scale_factor"]
    
    # Ensure dimensions are divisible by VAE scale factor
    adjusted_width = (width // vae_scale) * vae_scale
    adjusted_height = (height // vae_scale) * vae_scale
    
    # Ensure minimum dimensions
    min_size = vae_scale * 8  # Minimum 64 pixels
    adjusted_width = max(adjusted_width, min_size)
    adjusted_height = max(adjusted_height, min_size)
    
    return adjusted_width, adjusted_height


# Additional image generation utilities
class ImageGenerationHelper:
    """Helper class for image generation operations"""
    
    @staticmethod
    def prepare_image_metadata(
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
        lora_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Prepare metadata for generated images
        
        Args:
            Various generation parameters
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "model": "FramePack-I2V",
            "mode": "image_generation"
        }
        
        if lora_info:
            metadata["lora_paths"] = lora_info.get("paths", [])
            metadata["lora_scales"] = lora_info.get("scales", [])
            
        return metadata
    
    @staticmethod
    def calculate_memory_requirements(
        batch_size: int,
        width: int,
        height: int,
        dtype: torch.dtype = torch.float16
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for image generation
        
        Args:
            batch_size: Number of images
            width: Image width
            height: Image height
            dtype: Data type for tensors
            
        Returns:
            Dictionary with memory estimates in GB
        """
        # Calculate latent size
        latent_shape = get_image_latent_shape(batch_size, height, width)
        latent_elements = 1
        for dim in latent_shape:
            latent_elements *= dim
            
        # Bytes per element
        bytes_per_element = 2 if dtype == torch.float16 else 4
        
        # Estimate memory usage
        latent_memory = (latent_elements * bytes_per_element) / (1024**3)  # GB
        
        # Rough estimates for other components
        text_encoder_memory = 0.5  # GB
        vae_memory = 1.0  # GB
        transformer_memory = 8.0  # GB (for the base model)
        
        # Working memory estimate (buffers, gradients if training, etc.)
        working_memory = latent_memory * 4
        
        return {
            "latent_memory": latent_memory,
            "text_encoder_memory": text_encoder_memory,
            "vae_memory": vae_memory,
            "transformer_memory": transformer_memory,
            "working_memory": working_memory,
            "total_estimated": (
                latent_memory + text_encoder_memory + 
                vae_memory + transformer_memory + working_memory
            )
        }