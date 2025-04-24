# API改修計画: /result エンドポイントでの入力画像サムネイル返却

## 1. 目的

FastAPIアプリケーションの `/result/{job_id}` エンドポイントを改修し、生成された動画ファイルのダウンロードURLに加えて、ジョブに使用された**入力画像のサムネイル画像データ（Base64エンコード）**もレスポンスに含めるようにする。

## 2. 改修内容

### 2.1. `worker.py`

* **サムネイル生成処理の追加:**
  * `worker` 関数の入力画像読み込み後 (`Image.open(input_image_path)`) に、サムネイル画像を生成する処理を追加する。
  * サムネイル画像は `PIL` を使用してリサイズし、ファイル名 `thumb_{job_id}.jpg` として `settings.TEMP_QUEUE_IMAGES_DIR` ディレクトリに保存する。
* **サムネイルパス保存依頼の追加:**
  * ジョブステータスを `"processing"` に更新する `queue_manager.update_job_status` 呼び出し時に、`thumbnail` 引数に生成したサムネイル画像のフルパスを渡す。

### 2.2. `queue_manager.py`

* **変更なし:** 既存の `QueuedJob` データクラスの `thumbnail` フィールドと `update_job_status` 関数の `thumbnail` 引数をそのまま利用する。

### 2.3. `api.py`

* **レスポンスモデルの定義:**
  * `/result/{job_id}` 用の新しいPydanticレスポンスモデル (例: `ResultResponse`) を定義する。
  * モデルには、動画ダウンロードURL (`video_url: str`) とサムネイルのBase64データ (`thumbnail_base64: str`) を含める。
* **`/result/{job_id}` エンドポイントの改修:**
  * レスポンスモデルを `FileResponse` から定義した `ResultResponse` に変更する。
  * `queue_manager.get_job_by_id` でジョブ情報を取得する。
  * `job.thumbnail` に保存されているサムネイル画像のパスを取得する。
  * サムネイルファイルを読み込み、Base64エンコードする。
  * 適切なData URIスキーム (`data:image/jpeg;base64,...`) を作成する。
  * 動画ダウンロード用URL (`/download/video/{job_id}`) とBase64エンコードされたサムネイルデータを含むJSONレスポンスを返す。
* **`/download/video/{job_id}` エンドポイントの新設:**
  * 指定された `job_id` に対応する動画ファイル (`outputs/{job_id}.mp4`) を `FileResponse` で返すエンドポイントを追加する。

## 3. 処理フロー図 (Mermaid)

```mermaid
sequenceDiagram
    participant Client
    participant API (api.py)
    participant Worker (worker.py)
    participant QueueManager (queue_manager.py)
    participant FileSystem

    Client->>+API: POST /generate (画像アップロード)
    API->>+QueueManager: add_to_queue(image, ...)
    QueueManager->>FileSystem: save_image_to_temp (queue_image_{job_id}.jpg)
    QueueManager-->>-API: job_id
    API-->>-Client: {job_id: ...}

    Note over Worker, QueueManager: Background Worker picks up job
    Worker->>+QueueManager: get_next_job()
    QueueManager->>FileSystem: load_queue_from_file()
    QueueManager-->>-Worker: job
    Worker->>FileSystem: Image.open(job.image_path)
    Worker->>Worker: Generate Thumbnail (thumb_{job_id}.jpg)
    Worker->>FileSystem: Save Thumbnail (temp_queue_images/thumb_{job_id}.jpg)
    Worker->>+QueueManager: update_job_status(job_id, "processing", thumbnail_path) # サムネイルパスを保存
    QueueManager->>FileSystem: load_queue_from_file()
    QueueManager->>FileSystem: save_queue() (update status & thumbnail path)
    QueueManager-->>-Worker: True
    Note over Worker: Video Generation Process...
    Worker->>FileSystem: Save Video (outputs/{job_id}.mp4)
    Worker->>+QueueManager: update_job_status(job_id, "completed")
    QueueManager->>FileSystem: load_queue_from_file()
    QueueManager->>FileSystem: save_queue() (update status)
    QueueManager-->>-Worker: True

    Client->>+API: GET /result/{job_id}
    API->>+QueueManager: get_job_by_id(job_id)
    QueueManager->>FileSystem: load_queue_from_file()
    QueueManager-->>-API: job (with thumbnail path)
    API->>FileSystem: Read thumbnail file (job.thumbnail)
    API->>API: Encode thumbnail to Base64
    API->>API: Construct video_url (pointing to download endpoint)
    API-->>-Client: JSON { video_url: "/download/video/...", thumbnail_base64: "data:image/jpeg;base64,..." }

    Client->>+API: GET /download/video/{job_id}
    API->>FileSystem: Check outputs/{job_id}.mp4 exists
    API-->>-Client: FileResponse (video/mp4)

    # Note: Thumbnail download endpoint is no longer needed
