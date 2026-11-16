# FramePack-FastAPI 画像生成機能統合計画

## 概要
`oneframe_ichi.py`の高度な画像生成機能を現在のFramePack-FastAPIプロジェクトに統合します。これにより、動画生成に加えて、高品質な単一画像生成、バッチ処理、画像転写（kisekaeichi）機能を提供します。

## 1. 新しいAPIエンドポイント

### 1.1 画像生成エンドポイント
```
POST /api/generate-image
- 単一の高品質画像を生成
- Request Body:
  {
    "prompt": string,
    "negative_prompt": string (optional),
    "seed": integer (optional),
    "steps": integer (default: 30),
    "cfg": float (default: 1.0),
    "width": integer (default: 1216),
    "height": integer (default: 704),
    "lora_paths": array[string] (optional),
    "lora_scales": array[float] (optional)
  }
```

### 1.2 バッチ画像生成エンドポイント
```
POST /api/batch-images
- 複数の画像を一括生成
- Request Body:
  {
    "prompts": array[string],
    "negative_prompt": string (optional),
    "seeds": array[integer] (optional),
    "batch_size": integer (default: 4),
    "steps": integer (default: 30),
    "cfg": float (default: 1.0),
    "width": integer (default: 1216),
    "height": integer (default: 704),
    "lora_paths": array[string] (optional),
    "lora_scales": array[float] (optional)
  }
```

### 1.3 画像転写エンドポイント
```
POST /api/transfer-image
- Kisekaeichi機能：ある画像のスタイルを別の画像に転写
- Request Body:
  {
    "source_image": string (base64),
    "target_image": string (base64),
    "prompt": string,
    "negative_prompt": string (optional),
    "transfer_strength": float (default: 0.7),
    "seed": integer (optional),
    "steps": integer (default: 30),
    "cfg": float (default: 1.0)
  }
```

## 2. ファイル構成の変更

### 2.1 新規ファイル

#### `api/worker_image.py`
画像生成専用のワーカープロセス
```python
import torch
import numpy as np
from PIL import Image
import io
import base64
from typing import Optional, List, Dict, Any

from diffusers_helper.utils import save_image
from api.models import load_models
from api.queue_manager import QueueManager
from api.settings import get_settings

# RoPE関連の処理を実装
def rope_attention_batch(...):
    """RoPE attention batch processing for image generation"""
    pass

# 画像生成メイン処理
async def process_image_generation(job_id: str, job_data: dict, models: dict):
    """
    単一画像生成の処理
    """
    try:
        queue_manager = QueueManager()
        queue_manager.update_job_status(job_id, "processing")
        
        # パラメータの取得
        prompt = job_data["prompt"]
        negative_prompt = job_data.get("negative_prompt", "")
        seed = job_data.get("seed", -1)
        steps = job_data.get("steps", 30)
        cfg = job_data.get("cfg", 1.0)
        width = job_data.get("width", 1216)
        height = job_data.get("height", 704)
        
        # 画像生成処理
        # ... (詳細な実装)
        
        # 結果の保存
        output_path = f"outputs/images/{job_id}.png"
        save_image(generated_image, output_path, metadata=metadata)
        
        queue_manager.update_job_result(job_id, {"image_path": output_path})
        queue_manager.update_job_status(job_id, "completed")
        
    except Exception as e:
        queue_manager.update_job_status(job_id, "failed", str(e))

# バッチ処理
async def process_batch_images(job_id: str, job_data: dict, models: dict):
    """
    バッチ画像生成の処理
    """
    pass

# 画像転写処理
async def process_image_transfer(job_id: str, job_data: dict, models: dict):
    """
    Kisekaeichi画像転写の処理
    """
    pass
```

#### `api/image_models.py`
画像生成用の追加モデル設定
```python
from api.models import *

# 画像生成用の特別な設定
IMAGE_GENERATION_CONFIG = {
    "default_width": 1216,
    "default_height": 704,
    "max_batch_size": 8,
    "rope_batch_size": 4,
    "latent_channels": 16,
    "vae_scale_factor": 8,
    "time_scale_factor": 4,
}

# RoPE処理用の設定
def get_rope_config(batch_size: int, height: int, width: int):
    """RoPE処理の設定を取得"""
    return {
        "batch_size": batch_size,
        "height": height // 8,  # latent space
        "width": width // 8,
        "max_seq_length": (height // 8) * (width // 8),
        "embed_dim": 3072,  # transformer embed dim
    }
```

### 2.2 既存ファイルの修正

#### `api/models.py` の拡張
```python
# 既存のコードに追加

# 画像生成用の追加設定
IMAGE_GENERATION_MODELS = {
    "rope_processor": None,  # 後で初期化
    "latent_processor": None,
}

def load_image_generation_components():
    """画像生成用の追加コンポーネントをロード"""
    global IMAGE_GENERATION_MODELS
    
    # RoPE処理用のコンポーネント
    # ... 実装
    
    return IMAGE_GENERATION_MODELS

# load_models()関数の最後に追加
def load_models():
    # ... 既存のコード ...
    
    # 画像生成コンポーネントの追加
    image_components = load_image_generation_components()
    loaded_models.update(image_components)
    
    return loaded_models
```

#### `api/api.py` の拡張
```python
from fastapi import File, UploadFile
from api.worker_image import process_image_generation, process_batch_images, process_image_transfer

# 画像生成用のリクエストモデル
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = -1
    steps: int = 30
    cfg: float = 1.0
    width: int = 1216
    height: int = 704
    lora_paths: Optional[List[str]] = None
    lora_scales: Optional[List[float]] = None

class BatchImageRequest(BaseModel):
    prompts: List[str]
    negative_prompt: Optional[str] = ""
    seeds: Optional[List[int]] = None
    batch_size: int = 4
    steps: int = 30
    cfg: float = 1.0
    width: int = 1216
    height: int = 704
    lora_paths: Optional[List[str]] = None
    lora_scales: Optional[List[float]] = None

class ImageTransferRequest(BaseModel):
    source_image: str  # base64
    target_image: str  # base64
    prompt: str
    negative_prompt: Optional[str] = ""
    transfer_strength: float = 0.7
    seed: Optional[int] = -1
    steps: int = 30
    cfg: float = 1.0

# 新しいエンドポイント
@app.post("/api/generate-image")
async def generate_image(request: ImageGenerationRequest):
    job_id = str(uuid.uuid4())
    job_data = {
        "type": "image",
        "data": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
    queue_manager.add_job(job_id, job_data)
    
    # バックグラウンドで処理開始
    asyncio.create_task(process_image_generation(job_id, job_data, models))
    
    return {"job_id": job_id}

@app.post("/api/batch-images")
async def batch_images(request: BatchImageRequest):
    job_id = str(uuid.uuid4())
    job_data = {
        "type": "batch_image",
        "data": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
    queue_manager.add_job(job_id, job_data)
    
    # バックグラウンドで処理開始
    asyncio.create_task(process_batch_images(job_id, job_data, models))
    
    return {"job_id": job_id}

@app.post("/api/transfer-image")
async def transfer_image(request: ImageTransferRequest):
    job_id = str(uuid.uuid4())
    job_data = {
        "type": "image_transfer",
        "data": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
    queue_manager.add_job(job_id, job_data)
    
    # バックグラウンドで処理開始
    asyncio.create_task(process_image_transfer(job_id, job_data, models))
    
    return {"job_id": job_id}
```

#### `api/queue_manager.py` の拡張
```python
# ジョブタイプの追加
JOB_TYPES = ["video", "image", "batch_image", "image_transfer"]

class QueuedJob(BaseModel):
    id: str
    type: str  # "video", "image", "batch_image", "image_transfer"
    data: Dict[str, Any]
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None
    step: Optional[int] = None
    total: Optional[int] = None
    info: Optional[str] = None
    preview: Optional[str] = None  # base64 encoded preview
    batch_progress: Optional[Dict[str, Any]] = None  # for batch jobs
```

## 3. 実装優先順位

### Phase 1: 基本的な単一画像生成（1-2日）
1. `worker_image.py`の基本実装
2. `/api/generate-image`エンドポイントの実装
3. 既存のモデルローダーとの統合
4. 基本的なテスト

### Phase 2: バッチ処理（2-3日）
1. バッチ処理ロジックの実装
2. `/api/batch-images`エンドポイントの実装
3. 進捗トラッキングの拡張
4. メモリ管理の最適化

### Phase 3: 画像転写機能（3-4日）
1. Kisekaeichi機能の実装
2. `/api/transfer-image`エンドポイントの実装
3. マスク処理とブレンディングロジック
4. 高度なパラメータチューニング

### Phase 4: 最適化とテスト（2-3日）
1. パフォーマンス最適化
2. エラーハンドリングの強化
3. 包括的なテストスイート
4. ドキュメンテーション

## 4. 技術的な考慮事項

### 4.1 メモリ管理
- 動画生成と画像生成でモデルを共有
- 低VRAMモードでの適切なモデルスワップ
- バッチ処理時のメモリ効率化

### 4.2 並行処理
- 動画生成ジョブと画像生成ジョブの優先順位管理
- リソースの公平な配分
- キューシステムの拡張

### 4.3 互換性
- 既存のAPIとの後方互換性を維持
- 統一されたジョブステータス管理
- 共通のエラーハンドリング

## 5. 期待される成果

1. **統合プラットフォーム**: 動画と画像の両方を生成できる統一APIプラットフォーム
2. **効率的なリソース利用**: モデルの共有によるメモリ効率の向上
3. **高度な機能**: バッチ処理、画像転写などの高度な機能
4. **スケーラビリティ**: 将来的な機能拡張への対応

## 6. リスクと対策

### リスク
1. メモリ不足による処理の失敗
2. 動画生成と画像生成の競合
3. APIの複雑化

### 対策
1. 適切なメモリ管理とモデルスワップ
2. ジョブスケジューリングの最適化
3. 明確なAPIドキュメンテーション

## 7. 次のステップ

1. この計画のレビューと承認
2. 開発環境のセットアップ
3. Phase 1の実装開始
4. 定期的な進捗確認とフィードバック