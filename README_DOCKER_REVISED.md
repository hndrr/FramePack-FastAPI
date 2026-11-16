# FramePack-FastAPI Docker Guide (現実版)

## 🎯 重要なお知らせ

**CPU版での推論は現実的ではありません！** HunyuanVideoのような大型言語モデルは、CPUでは以下の理由で実用的ではありません：

- **メモリ要件**: 60GB以上のRAMが必要
- **処理時間**: CPU推論は数時間〜数日かかる
- **モデルサイズ**: 40GB以上のモデルファイル

## 🐳 Docker構成の説明

### 1. **開発用サーバー** (`Dockerfile`)
- **用途**: API開発・テスト専用
- **含まれるもの**: FastAPI, 基本的な依存関係のみ
- **モデル**: なし（軽量、数MB）
- **機能**: APIエンドポイントのテスト、開発

### 2. **GPU推論サーバー** (`Dockerfile.gpu`)
- **用途**: 実際の動画・画像生成
- **要件**: NVIDIA GPU（24GB+ VRAM推奨）
- **モデル**: 全ての必要なモデル（40GB+）
- **機能**: フル機能の推論

## 🚀 使い方

### 開発・テスト用（軽量版）

```bash
# 開発サーバー起動（モデルなし）
./docker-run.sh

# または
docker-compose up

# APIテストはできるが、推論は503エラーを返す
curl http://localhost:8000/docs  # ✅ 動作
curl -X POST http://localhost:8000/api/generate-image  # ❌ 503エラー
```

### 本格運用（GPU必須）

```bash
# GPU必須: NVIDIA GPU + 24GB+ VRAM
./docker-run.sh --gpu

# または
docker-compose --profile gpu up
docker-compose -f docker-compose.gpu.yml up

# フル機能利用可能
curl http://localhost:8001/docs  # ✅ 動作
curl -X POST http://localhost:8001/api/generate-image  # ✅ 実際に画像生成
```

## ⚡ 現実的な運用方法

### 1. **個人開発者**
```bash
# ローカル開発（モデルなし）
./docker-run.sh

# GPU搭載マシンで推論テスト
./docker-run.sh --gpu
```

### 2. **チーム開発**
```bash
# 開発チーム: 軽量APIサーバー
docker-compose up framepack-api-dev

# GPU担当者: 推論サーバー
docker-compose --profile gpu up framepack-api-gpu
```

### 3. **本番環境**
```bash
# クラウドGPUインスタンス（AWS p3.2xlarge等）
docker-compose -f docker-compose.gpu.yml up -d

# ロードバランサー経由でアクセス
```

## 💻 システム要件

### 開発用サーバー
- **RAM**: 1GB以下
- **ストレージ**: 1GB以下
- **CPU**: 任意
- **起動時間**: 10秒以下

### GPU推論サーバー
- **GPU**: NVIDIA RTX 3090/4090, A100, H100等
- **VRAM**: 24GB以上（推奨: 48GB以上）
- **RAM**: 64GB以上
- **ストレージ**: 100GB以上の空き容量
- **起動時間**: 2-5分（モデルダウンロード時は30分+）

## 🔧 Docker構成詳細

### ポート設定
- **開発サーバー**: `localhost:8000`
- **GPU推論サーバー**: `localhost:8001`

### ボリューム
```yaml
volumes:
  - ./outputs:/app/outputs              # 生成結果
  - ./loras:/app/loras                  # LoRAファイル
  - ./hf_download:/app/hf_download      # モデルキャッシュ（GPU版のみ）
  - ./temp_queue_images:/app/temp_queue_images  # 一時ファイル
```

### 環境変数
```yaml
# 開発版
MODELS_DISABLED=true
PYTHONUNBUFFERED=1

# GPU版
NVIDIA_VISIBLE_DEVICES=all
HF_HOME=/app/hf_download
TRANSFORMERS_CACHE=/app/hf_download
```

## 🛠️ トラブルシューティング

### 開発サーバー

#### Q: APIが503エラーを返す
**A:** 正常です。開発サーバーはモデルがないため推論できません。

#### Q: どのエンドポイントがテスト可能？
**A:** 以下のエンドポイントは動作します：
- `GET /docs` - API ドキュメント
- `GET /queue` - キュー状態
- `GET /worker/status` - ワーカー状態
- `GET /loras` - LoRAリスト

### GPU推論サーバー

#### Q: "CUDA out of memory"エラー
**A:** VRAMが不足しています。
```bash
# VRAM使用量確認
docker exec -it <container> nvidia-smi

# より小さなバッチサイズを使用
# または、より多くのVRAMを持つGPUを使用
```

#### Q: モデルダウンロードが遅い
**A:** 初回起動時は40GB+のモデルをダウンロードします。
```bash
# 事前ダウンロード（推奨）
mkdir -p hf_download
# モデルを事前に配置するか、高速回線での初回起動を推奨
```

#### Q: コンテナが起動しない
**A:** NVIDIA Dockerが正しく設定されているか確認：
```bash
# NVIDIA Docker確認
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# Docker Composeでのprofile指定確認
docker-compose --profile gpu config
```

## 🎯 おすすめワークフロー

### 1. 開発段階
```bash
# 軽量サーバーでAPI開発
./docker-run.sh
```

### 2. テスト段階
```bash
# GPU環境で機能テスト
./docker-run.sh --gpu
```

### 3. 本番段階
```bash
# 本番GPUサーバーにデプロイ
docker-compose -f docker-compose.gpu.yml up -d
```

## 💡 コスト最適化のヒント

1. **開発時**: 開発版のみ使用（コスト: $0/月）
2. **テスト時**: クラウドGPUを時間課金で使用（コスト: $1-3/時間）
3. **本番時**: 専用GPUサーバーまたはGPUクラウド（コスト: $500-2000/月）

## 🚨 注意事項

- **CPU版は推論不可**: 開発・テスト専用です
- **GPU版は重い**: 初回起動に時間がかかります
- **VRAM要件**: 24GB未満のGPUでは動作しない可能性があります
- **ストレージ**: モデルで50GB以上の容量を消費します

この設定により、現実的で実用的なDocker環境が構築できます！