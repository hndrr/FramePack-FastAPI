# FramePack-FastAPI Dockerfile 作成計画

## 概要

このドキュメントは、`FramePack-FastAPI` プロジェクトの Docker イメージを構築するための Dockerfile 作成計画を記述します。GPU (CUDA 12.6) 利用、Hugging Face Hub へのアクセス、FastAPI サーバーの実行を考慮します。

## 計画詳細

1. **ベースイメージの選定:**
    * Python 3.10 と CUDA 12.6 に対応する公式イメージを選択します。
    * 候補: `nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04` (または同等の機能を持つイメージ)
    * 理由: Python バージョン、CUDA 要件を満たし、`apt-get` によるシステムライブラリのインストールが可能です。

2. **システム依存ライブラリのインストール:**
    * `opencv-python-contrib`, `av`, モデルダウンロード (`git`) に必要なライブラリをインストールします。
    * コマンド: `apt-get update && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git && rm -rf /var/lib/apt/lists/*`
    * 理由: アプリケーションの実行とモデル取得に必要な依存関係を解決します。`--no-install-recommends` と `rm -rf /var/lib/apt/lists/*` でイメージサイズを削減します。

3. **Python 環境のセットアップ:**
    * `pip` を最新バージョンにアップグレードします。
    * `requirements.txt` をコンテナにコピーします。
    * **GPU 対応 PyTorch のインストール:** `README.md` に記載の CUDA 12.6 対応コマンドを実行します。
        * コマンド: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
    * **残りの依存関係のインストール:** `requirements.txt` から `torch` と `torchvision` を除外してインストールします。
        * `requirements.txt` を編集するか、`grep -vE '^torch|^torchvision'` などでフィルタリングしてインストールします。
        * コマンド例 (フィルタリング): `grep -vE '^torch|^torchvision' requirements.txt | pip install --no-cache-dir -r /dev/stdin`
        * 理由: `requirements.txt` に記載の CPU 版 PyTorch を避け、指定された GPU 版をインストールします。`--no-cache-dir` でイメージサイズを削減します。

4. **アプリケーションコードのコピー:**
    * `.dockerignore` ファイルを作成し、不要なファイルやディレクトリ (`.git`, `venv/`, `__pycache__`, `*.7z`, `temp_queue_images/`, `tests/`, `*.md` など) を指定して、ビルドコンテキストとイメージサイズを削減します。
    * プロジェクト全体 (`.`) をコンテナ内の作業ディレクトリ `/app` にコピーします。
    * コマンド:

        ```dockerfile
        COPY .dockerignore .dockerignore
        COPY . /app
        ```

5. **Hugging Face Hub 認証:**
    * 認証トークンは Dockerfile に含めず、コンテナ実行時に環境変数 `HUGGING_FACE_HUB_TOKEN` として渡すことを想定します。
    * 理由: セキュリティのベストプラクティスに従います。Hugging Face ライブラリは通常、この環境変数を自動的に検出します。

6. **ポートの公開:**
    * FastAPI アプリケーションが使用するポート `8080` を公開します。
    * コマンド: `EXPOSE 8080`

7. **作業ディレクトリの設定:**
    * コンテナ内の作業ディレクトリを `/app` に設定します。
    * コマンド: `WORKDIR /app`

8. **起動コマンドの設定:**
    * コンテナ起動時に FastAPI サーバーを実行するコマンドを設定します。
    * コマンド: `CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]`

## 計画の視覚化 (Mermaid)

```mermaid
graph TD
    A[ベースイメージ選定 (Python 3.10 + CUDA 12.6)] --> B(システム依存ライブラリ インストール (ffmpeg, libsm6, libxext6, git));
    B --> C(Python 環境セットアップ (pip upgrade));
    C --> D(GPU対応PyTorchインストール (cu126));
    D --> E(requirements.txt インストール (torch除く, --no-cache-dir));
    E --> F(アプリケーションコード コピー (`.` -> `/app`, .dockerignore));
    F --> G(Hugging Face Hub 認証設定 (環境変数想定));
    G --> H(ポート公開: 8080);
    H --> I(作業ディレクトリ設定 (/app));
    I --> J(起動コマンド設定 (uvicorn));
```

## 次のステップ

この計画に基づき、Code モードで `Dockerfile` と `.dockerignore` ファイルを作成します。
