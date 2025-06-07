# FramePack-FastAPI Docker Guide

このガイドでは、FramePack-FastAPIをDockerで実行する方法を説明します。

## 🐳 Docker構成

### 利用可能なDockerfile

1. **Dockerfile** - CPU版（軽量、推論のみ）
2. **Dockerfile.gpu** - GPU版（CUDA対応、高性能）

### Docker Compose設定

1. **docker-compose.yml** - CPU版用
2. **docker-compose.gpu.yml** - GPU版用

## 🚀 クイックスタート

### CPU版を起動
```bash
# 自動ビルド＆起動
./docker-run.sh

# または手動で
docker-compose up
```

### GPU版を起動
```bash
# 自動ビルド＆起動（GPU必須）
./docker-run.sh --gpu

# または手動で
docker-compose -f docker-compose.gpu.yml up
```

## 🔨 手動ビルド

### CPUイメージをビルド
```bash
./docker-build.sh
# または
docker build -f Dockerfile -t framepack-fastapi:latest .
```

### GPUイメージをビルド
```bash
./docker-build.sh --gpu
# または
docker build -f Dockerfile.gpu -t framepack-fastapi-gpu:latest .
```

## 🏃‍♂️ 実行方法

### 1. Docker Composeを使用（推奨）

#### CPU版
```bash
docker-compose up -d  # バックグラウンド実行
docker-compose logs -f  # ログを表示
```

#### GPU版
```bash
docker-compose -f docker-compose.gpu.yml up -d
docker-compose -f docker-compose.gpu.yml logs -f
```

### 2. 直接Dockerコマンドを使用

#### CPU版
```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/temp_queue_images:/app/temp_queue_images \
  -v $(pwd)/loras:/app/loras \
  -v $(pwd)/hf_download:/app/hf_download \
  framepack-fastapi:latest
```

#### GPU版
```bash
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/temp_queue_images:/app/temp_queue_images \
  -v $(pwd)/loras:/app/loras \
  -v $(pwd)/hf_download:/app/hf_download \
  framepack-fastapi-gpu:latest
```

## 🔧 カスタマイズ

### ポート変更
```bash
# ポート8080で起動
./docker-run.sh --port 8080

# または環境変数で設定
export PORT=8080
docker-compose up
```

### 環境変数

重要な環境変数：

```bash
# API設定
API_HOST=0.0.0.0
API_PORT=8000

# モデルキャッシュ
HF_HOME=/app/hf_download
TRANSFORMERS_CACHE=/app/hf_download

# GPU設定（GPU版のみ）
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
```

### ボリューム設定

永続化されるデータ：

- `./outputs` - 生成された動画・画像
- `./temp_queue_images` - キュー用一時画像
- `./loras` - LoRAファイル
- `./hf_download` - Hugging Faceモデルキャッシュ
- `./job_queue.json` - ジョブキュー状態

## 📋 システム要件

### CPU版
- **RAM**: 最小8GB、推奨16GB以上
- **ストレージ**: 最小20GB の空き容量
- **CPU**: マルチコア推奨

### GPU版
- **GPU**: NVIDIA GPU（CUDA 12.1対応）
- **VRAM**: 最小8GB、推奨24GB以上
- **RAM**: 最小16GB、推奨32GB以上
- **ストレージ**: 最小50GB の空き容量
- **NVIDIA Docker**: インストール済み

## 🛠️ トラブルシューティング

### よくある問題

#### 1. GPU版でGPUが認識されない
```bash
# NVIDIA Dockerの確認
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# GPU対応確認
docker-compose -f docker-compose.gpu.yml exec framepack-api-gpu python -c "import torch; print(torch.cuda.is_available())"
```

#### 2. メモリ不足エラー
```bash
# Dockerのメモリ制限を確認・調整
docker stats

# Swapファイルを有効化（Linux）
sudo swapon --show
```

#### 3. モデルダウンロードが遅い/失敗する
```bash
# Hugging Face キャッシュを事前ダウンロード
mkdir -p hf_download
docker run -v $(pwd)/hf_download:/app/hf_download framepack-fastapi-gpu:latest python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('hunyuanvideo-community/HunyuanVideo', cache_dir='/app/hf_download')
"
```

#### 4. ポート競合
```bash
# 使用中のポートを確認
netstat -tlnp | grep :8000

# 別のポートを使用
docker-compose up -e API_PORT=8080
```

### ログの確認

```bash
# サービスログ
docker-compose logs framepack-api

# リアルタイムログ
docker-compose logs -f framepack-api

# 特定コンテナのログ
docker logs <container_id>
```

### パフォーマンス監視

```bash
# リソース使用量
docker stats

# GPU使用量（GPU版）
docker-compose -f docker-compose.gpu.yml exec framepack-api-gpu nvidia-smi
```

## 🌐 API アクセス

コンテナ起動後、以下のURLでアクセス可能：

- **API**: http://localhost:8000
- **API ドキュメント**: http://localhost:8000/docs
- **動画生成**: http://localhost:8000/generate
- **画像生成**: http://localhost:8000/api/generate-image

## 🔄 更新とメンテナンス

### イメージの更新
```bash
# 最新コードでリビルド
./docker-build.sh --tag latest

# 古いイメージを削除
docker image prune -f
```

### データのバックアップ
```bash
# 生成データをバックアップ
tar -czf framepack-backup-$(date +%Y%m%d).tar.gz outputs/ temp_queue_images/ loras/ hf_download/
```

### コンテナの停止と削除
```bash
# サービス停止
docker-compose down

# データボリュームも含めて削除
docker-compose down -v

# 完全クリーンアップ
docker system prune -af
```