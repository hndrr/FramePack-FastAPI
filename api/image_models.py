"""
Image generation specific models and configurations
Enhanced with performance optimization utilities
"""
from typing import Dict, Any, Tuple, List
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
    "text_encoding_cache_size": 50,  # Max cached text encodings
    "memory_cleanup_threshold": 0.9,  # GPU memory threshold for cleanup
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


def check_gpu_memory() -> Dict[str, float]:
    """
    Check current GPU memory usage
    
    Returns:
        Dictionary with memory statistics in GB
    """
    if not torch.cuda.is_available():
        return {
            "allocated": 0.0,
            "reserved": 0.0,
            "free": 0.0,
            "total": 0.0
        }
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - allocated
    
    return {
        "allocated": allocated,
        "reserved": reserved,
        "free": free,
        "total": total
    }


def should_cleanup_memory(threshold: float = None) -> bool:
    """
    Check if memory cleanup is needed based on usage threshold
    
    Args:
        threshold: Memory usage threshold (0-1), defaults to config value
        
    Returns:
        True if cleanup is recommended
    """
    if threshold is None:
        threshold = IMAGE_GENERATION_CONFIG["memory_cleanup_threshold"]
    
    memory_stats = check_gpu_memory()
    if memory_stats["total"] == 0:
        return False
    
    usage_ratio = memory_stats["allocated"] / memory_stats["total"]
    return usage_ratio >= threshold


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
        lora_info: Dict[str, Any] = None,
        use_fp8: bool = False,
        use_vae_cache: bool = False,
        sampling_mode: str = "dpm-solver++",
        transformer_model: str = "base"
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
            "model": f"FramePack-HunyuanVideo ({transformer_model})",
            "mode": "image_generation",
            "sampling_mode": sampling_mode,
            "use_fp8": use_fp8,
            "use_vae_cache": use_vae_cache
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
        dtype: torch.dtype = torch.float16,
        use_fp8: bool = False,
        use_vae_cache: bool = False
    ) -> Dict[str, float]:
        """
        Estimate memory requirements for image generation
        
        Args:
            batch_size: Number of images
            width: Image width
            height: Image height
            dtype: Data type for tensors
            use_fp8: Whether FP8 optimization is used
            use_vae_cache: Whether VAE caching is used
            
        Returns:
            Dictionary with memory estimates in GB
        """
        # Calculate latent size
        latent_shape = get_image_latent_shape(batch_size, height, width)
        latent_elements = 1
        for dim in latent_shape:
            latent_elements *= dim
            
        # Bytes per element
        if use_fp8:
            bytes_per_element = 1  # FP8
        elif dtype == torch.float16:
            bytes_per_element = 2
        else:
            bytes_per_element = 4
        
        # Estimate memory usage
        latent_memory = (latent_elements * bytes_per_element) / (1024**3)  # GB
        
        # Rough estimates for other components
        text_encoder_memory = 0.5  # GB
        vae_memory = 1.0 if not use_vae_cache else 0.5  # GB (reduced with caching)
        transformer_memory = 8.0 if not use_fp8 else 4.0  # GB (reduced with FP8)
        
        # Working memory estimate (buffers, gradients if training, etc.)
        working_memory = latent_memory * 3 if not use_vae_cache else latent_memory * 2
        
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
    
    @staticmethod
    def optimize_batch_size(
        width: int,
        height: int,
        available_memory_gb: float,
        use_fp8: bool = False,
        use_vae_cache: bool = False
    ) -> int:
        """
        Calculate optimal batch size based on available memory
        
        Args:
            width: Image width
            height: Image height
            available_memory_gb: Available GPU memory in GB
            use_fp8: Whether FP8 optimization is used
            use_vae_cache: Whether VAE caching is used
            
        Returns:
            Recommended batch size
        """
        # Start with batch size 1 and increase until memory limit
        for batch_size in range(1, IMAGE_GENERATION_CONFIG["max_batch_size"] + 1):
            memory_req = ImageGenerationHelper.calculate_memory_requirements(
                batch_size, width, height, 
                dtype=torch.float16,
                use_fp8=use_fp8,
                use_vae_cache=use_vae_cache
            )
            
            if memory_req["total_estimated"] > available_memory_gb * 0.9:  # 90% safety margin
                return max(1, batch_size - 1)
        
        return IMAGE_GENERATION_CONFIG["max_batch_size"]


class ImageGenerationOptimizer:
    """Advanced optimization utilities for image generation"""
    
    @staticmethod
    def get_optimal_settings(
        width: int,
        height: int,
        high_vram: bool = False
    ) -> Dict[str, Any]:
        """
        Get optimal settings based on resolution and VRAM
        
        Args:
            width: Target width
            height: Target height
            high_vram: Whether high VRAM mode is enabled
            
        Returns:
            Dictionary of recommended settings
        """
        memory_stats = check_gpu_memory()
        available_memory = memory_stats["free"]
        
        # Base recommendations
        settings = {
            "use_fp8": torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            "use_vae_cache": True,
            "gpu_memory_preservation": 10.0 if high_vram else 6.0,
            "text_encoding_cache": True,
            "clear_memory_steps": 5 if not high_vram else 10,
        }
        
        # Adjust batch size
        if high_vram:
            settings["batch_size"] = min(8, IMAGE_GENERATION_CONFIG["max_batch_size"])
        else:
            settings["batch_size"] = ImageGenerationHelper.optimize_batch_size(
                width, height, available_memory,
                use_fp8=settings["use_fp8"],
                use_vae_cache=settings["use_vae_cache"]
            )
        
        # Adjust based on resolution
        pixels = width * height
        if pixels > 1920 * 1080:  # Over 1080p
            settings["use_vae_cache"] = True
            settings["batch_size"] = min(settings["batch_size"], 2)
            settings["gpu_memory_preservation"] = 8.0 if not high_vram else 12.0
        
        return settings
    
    @staticmethod
    def prepare_lora_configs(
        lora_paths: List[str],
        lora_scales: List[float],
        lora_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Prepare LoRA configurations for batch loading
        
        Args:
            lora_paths: List of LoRA paths
            lora_scales: List of LoRA scales
            lora_dir: Base directory for LoRAs
            
        Returns:
            List of LoRA configuration dictionaries
        """
        import os
        
        configs = []
        for i, lora_path in enumerate(lora_paths):
            full_path = os.path.join(lora_dir, lora_path) if not os.path.isabs(lora_path) else lora_path
            if os.path.exists(full_path):
                scale = lora_scales[i] if i < len(lora_scales) else 1.0
                configs.append({
                    "path": full_path,
                    "scale": scale,
                    "name": os.path.basename(lora_path)
                })
        
        return configs