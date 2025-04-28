# API プレビュー機能 実装計画

## 1. 目的

`api/` ディレクトリ内の FastAPI アプリケーションにおいて、`demo_gradio.py` と同様の動画生成中のプレビュー機能を実装する。これにより、API クライアントは生成プロセスの途中経過をリアルタイムで確認できるようになる。

## 2. 現状分析と課題

* **`demo_gradio.py` のプレビュー:**
  * `worker` 内の `sample_hunyuan` コールバックで `vae_decode_fake` を使用し、プレビュー画像を生成。
  * `AsyncStream` を介して Gradio UI にプレビュー画像を送信。
* **`api/worker.py` の現状:**
  * `sample_hunyuan` のコールバックは進捗テキスト更新とキャンセルチェックのみ。プレビュー生成・送信は未実装。
  * `worker` からクライアントへのリアルタイムデータ送信手段が直接はない。
* **API での実現課題:**
  * `api/worker.py` のコールバックでプレビュー画像を生成する必要がある。
  * 生成したプレビュー画像を API クライアントにリアルタイムで送信する仕組みが必要。
  * `vae_decode_fake` 関数の利用可否確認 (→ 確認済み、利用可能)。

## 3. 計画

### 3.1. 情報収集 (完了)

* `diffusers_helper/hunyuan.py` を確認し、`vae_decode_fake` 関数が存在することを確認した。

### 3.2. API 設計

* 既存の Server-Sent Events (SSE) エンドポイント `/stream/status/{job_id}` (`api.py` 内) を拡張し、プレビュー画像データも送信するようにする。
* SSE イベントのデータ構造に、オプションとして Base64 エンコードされたプレビュー画像 (`preview_image_base64`) を追加する。

    ```json
    {
      "job_id": "...",
      "status": "processing",
      "progress": 25.5,
      "progress_step": 5,
      "progress_total": 20,
      "progress_info": "Sampling...",
      "preview_image_base64": "data:image/jpeg;base64,..." // Optional
    }
    ```

### 3.3. 実装方針

* **`api/worker.py` の修正:**
  * `callback` 関数内で、`sample_hunyuan` から渡される中間潜在変数 (`d['denoised']`) を取得する。
  * `vae_decode_fake` を使用してプレビュー画像を生成する。
  * 生成した画像を JPEG 形式にエンコードし、Base64 文字列に変換する (`data:image/jpeg;base64,...` 形式)。
  * 変換した Base64 文字列を `queue_manager` の新しい関数 (例: `update_current_preview`) を呼び出してメモリ上のストアに一時保存する。
* **`api/queue_manager.py` の修正:**
  * プレビュー情報 (Base64 文字列) を一時的に保持するためのグローバルな辞書 (例: `current_previews = {}`) を追加する。
  * `worker.py` からプレビュー情報を受け取り、`current_previews` を更新する関数 (例: `update_current_preview(job_id, preview_base64)`) を追加する。
  * SSE ハンドラからプレビュー情報を取得する関数 (例: `get_current_preview(job_id)`) を追加する。
  * ジョブ完了時または失敗時に `current_previews` から該当ジョブのエントリを削除する処理を追加する (例: `clear_current_preview(job_id)`)。
  * **注意:** このプレビュー情報は揮発性であり、JSON キューファイル (`job_queue.json`) には保存しない。
* **`api/api.py` の修正:**
  * `/stream/status/{job_id}` の SSE `event_generator` 関数を修正する。
  * ジョブが `processing` 状態の場合、`queue_manager` の新しい関数 (例: `get_current_preview`) を呼び出して最新のプレビュー画像 Base64 文字列を取得する。
  * 取得した Base64 文字列を SSE イベントデータの `preview_image_base64` フィールドに含めてクライアントに送信する。

### 3.4. 処理フロー (Mermaid図)

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI (api.py)
    participant QueueManager (queue_manager.py)
    participant Worker (worker.py)
    participant Models (models.py / diffusers_helper)

    Client->>FastAPI: POST /generate (画像, プロンプト)
    FastAPI->>QueueManager: add_to_queue()
    QueueManager-->>FastAPI: job_id
    FastAPI-->>Client: {job_id: ...}

    Client->>FastAPI: GET /stream/status/{job_id} (SSE接続)
    FastAPI->>FastAPI: event_generator() 開始

    loop Worker Thread
        Worker->>QueueManager: get_next_job()
        QueueManager-->>Worker: job (or None)
        opt job is not None
            Worker->>QueueManager: update_job_status(job_id, "processing") # ファイル更新
            Worker->>Models: モデルロード/準備 ...
            Worker->>Models: sample_hunyuan(..., callback=callback_func)
            loop Sampling Steps
                Models->>Worker: callback_func(d) 呼び出し
                Worker->>Models: vae_decode_fake(d['denoised']) # プレビュー生成
                Models-->>Worker: preview_image_tensor
                Worker->>Worker: 画像をJPEG Base64に変換
                Worker->>QueueManager: update_current_preview(job_id, preview_base64) # メモリ更新
                Worker->>QueueManager: update_job_progress(...) # ファイル更新 (進捗のみ)
            end
            Models-->>Worker: generated_latents
            Worker->>Models: vae_decode() # 最終デコード
            Models-->>Worker: final_pixels
            Worker->>Worker: save_bcthw_as_mp4()
            Worker->>QueueManager: update_job_status(job_id, "completed") # ファイル更新
            Worker->>QueueManager: clear_current_preview(job_id) # メモリクリア
        end
    end

    loop SSE Connection (event_generator)
        FastAPI->>QueueManager: get_job_by_id(job_id) (ファイルから進捗取得)
        alt job is processing
            FastAPI->>QueueManager: get_current_preview(job_id) (メモリからプレビュー取得)
            QueueManager-->>FastAPI: preview_base64 (or None)
        end
        FastAPI->>Client: event: progress, data: {..., preview_image_base64: ...} # 進捗とプレビュー送信
        alt job is terminal
            FastAPI->>Client: event: status, data: {...} # 最終ステータス送信
            break
        end
        FastAPI->>FastAPI: asyncio.sleep(1)
    end
```

## 4. 懸念点

* **データ受け渡し:** `worker` スレッドと SSE ハンドラ (FastAPI の非同期コンテキスト) 間でのプレビューデータ受け渡し (`queue_manager` のメモリ上の辞書) が、スレッドセーフティやパフォーマンスの観点から問題ないか。高頻度更新時の競合やメモリ使用量に注意が必要。
* **パフォーマンス:** プレビュー画像の生成 (`vae_decode_fake`)、JPEG エンコード、Base64 エンコードが `worker` のコールバック内で実行されるため、全体の生成時間に影響を与える可能性がある。

## 5. 次のステップ

* この計画に基づき、`code` モードに切り替えて実装を開始する。
