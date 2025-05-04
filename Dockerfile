FROM nvidia/cuda:12.6.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 残りの依存関係のインストール (torch と torchvision を除外)
# requirements.txt に torch/torchvision がバージョン指定なしで含まれているため、grepで除外
RUN grep -vE '^torch$|^torchvision$' requirements.txt | pip install --no-cache-dir -r /dev/stdin

# 5. Hugging Face Hub 認証 (ビルド時引数として定義)
# ビルド時に --build-arg HUGGING_FACE_HUB_TOKEN=your_token で渡す
# または実行時に環境変数として渡す場合はこの ARG/ENV は不要
ARG HUGGING_FACE_HUB_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}

COPY .dockerignore .dockerignore
COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]