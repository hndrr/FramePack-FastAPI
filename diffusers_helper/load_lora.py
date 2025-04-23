from pathlib import Path
from typing import Optional
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers


def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors", lora_scale: float = 1.0):  # 追加: lora_scale 引数 (デフォルト 1.0)
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.
        lora_scale (float): The scale factor (weight) to apply to the LoRA layers. Defaults to 1.0. # 追加: 引数の説明

    """

    state_dict = _fetch_state_dict(
        lora_path,
        weight_name,
        True,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    )

    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)

    # 修正: scale 引数を追加して lora_scale を渡す
    transformer.load_lora_adapter(state_dict, network_alphas=None, scale=lora_scale)
    print(f"LoRA weights loaded successfully with scale {lora_scale}.")  # 修正: スケール値をログに出力
    return transformer
