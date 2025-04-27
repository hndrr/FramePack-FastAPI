# 動画連続再生機能 実装計画

## 概要

ローカルの特定ディレクトリ (`outputs`) に順次追加される動画ファイル (`.mp4`) を検知し、FastAPIのSSE (Server-Sent Events) を通じてクライアント (React想定) に通知する。クライアントは通知されたファイル名を元に動画をリクエストし、連続再生を行う。既存のFramePack API (`api/api.py`) に機能を追加し、他の機能への影響を最小限に抑える。

## 計画詳細

1. **設定更新 (`api/settings.py`):**
    * 監視対象ディレクトリパス `VIDEO_DIR` を定義する。デフォルトはプロジェクトルート下の `outputs` ディレクトリとする。環境変数 `VIDEO_DIR` が設定されていれば、その値を優先する。
    * 動画ファイル配信用エンドポイントのベースURL `VIDEO_BASE_URL` を `/videos/` として定義する (主にクライアント側の参考情報)。

2. **ファイル監視ロジック (`api/video_watcher.py` - 新規ファイル):**
    * `watchdog` ライブラリを使用する `VideoHandler` クラスを作成する。
    * `on_created` イベントハンドラを実装し、`.mp4` ファイルが作成された場合のみ、FastAPI側のSSEクライアントキューリスト (`sse_clients`) にファイル名を追加する。
    * 監視を開始/停止する関数 (`start_watcher`, `stop_watcher`) を作成する。
        * `start_watcher(path, clients)`: 指定されたパスを監視し、通知先のクライアントキューリストを受け取る。`watchdog.observers.Observer` インスタンスを初期化・開始し、そのインスタンスを返す。
        * `stop_watcher(observer)`: 受け取った `Observer` インスタンスを停止・結合する。

3. **FastAPIエンドポイント追加 (`api/api.py`):**
    * **グローバル変数:**
        * `sse_clients = []`: SSEクライアントごとの通知キュー (`asyncio.Queue` など) を保持するリスト。
        * `observer = None`: `watchdog` の `Observer` インスタンスを保持する変数。
    * **`/video_stream` (GET, SSE):**
        * 新しいクライアント接続時に、専用の通知キューを作成し `sse_clients` に追加する。
        * 非同期ジェネレータ関数を定義する。
            * 無限ループでクライアントの接続状態をチェックする。
            * キューから新しいファイル名を取得し、`data: {filename}\n\n` 形式で `yield` する。
            * クライアント切断時には、対応するキューを `sse_clients` から削除し、ループを終了する。
        * `StreamingResponse` で上記ジェネレータを返す (`media_type="text/event-stream"`)。
    * **`/videos/{filename}` (GET):**
        * `settings.VIDEO_DIR` とリクエストされた `filename` を結合して、動画ファイルのフルパスを構築する。
        * `os.path.exists` でファイルの存在を確認する。
        * 存在すれば `FileResponse` を使用して動画ファイル (`media_type="video/mp4"`) を返す。
        * 存在しなければ `HTTPException(status_code=404, detail="File not found")` を発生させる。
    * **`/videos` (GET):**
        * `settings.VIDEO_DIR` 内のファイルを `os.listdir` で取得する。
        * ファイル名が `.mp4` で終わるもののみをフィルタリングする。
        * フィルタリングされたファイル名のリストをJSON形式で返す。

4. **ライフサイクル管理 (`api/api.py` の `lifespan`):**
    * 既存の `lifespan` コンテキストマネージャを修正する。
    * **Startup:**
        * `video_watcher.start_watcher(settings.VIDEO_DIR, sse_clients)` を呼び出し、返された `Observer` インスタンスをグローバル変数 `observer` に格納する。
    * **Shutdown:**
        * グローバル変数 `observer` が `None` でなければ、`video_watcher.stop_watcher(observer)` を呼び出してファイル監視プロセスを安全に停止する。

5. **依存関係:**
    * `watchdog` ライブラリが必要となるため、プロジェクトの依存関係ファイル (`requirements.txt` や `pyproject.toml` など) に `watchdog` を追加する。

## Mermaid図

```mermaid
graph TD
    subgraph FastAPI Backend (api/api.py)
        A[Client connects to /video_stream] --> B{Create SSE queue (e.g., asyncio.Queue)};
        B --> C[Add queue to global sse_clients list];
        C --> D[Start SSE generation loop (async def)];
        D -- New filename in queue --> E[yield f"data: {filename}\n\n"];
        D -- Client disconnects --> F[Remove queue from sse_clients & break loop];

        G[Client requests /videos/{filename}] --> H{Build file path using settings.VIDEO_DIR};
        H -- os.path.exists is True --> I[Return FileResponse(path, media_type="video/mp4")];
        H -- os.path.exists is False --> J[Raise HTTPException(404)];

        K[Client requests /videos] --> L{os.listdir(settings.VIDEO_DIR)};
        L --> M[Filter for .mp4 files, return JSON list];

        N[lifespan startup] --> O[observer = video_watcher.start_watcher(VIDEO_DIR, sse_clients)];
        P[lifespan shutdown] --> Q[if observer: video_watcher.stop_watcher(observer)];
    end

    subgraph File System Watcher (api/video_watcher.py - New File)
        R[Watchdog Observer monitors VIDEO_DIR] -- New .mp4 created --> S[VideoHandler.on_created];
        S --> T{Get filename};
        T --> U[Add filename to all queues in sse_clients list];
        V[start_watcher(path, clients)] --> W[Initialize Observer & Handler, observer.start(), return observer];
        X[stop_watcher(observer)] --> Y[observer.stop(), observer.join()];
    end

    subgraph React Frontend (Out of scope)
        Z[Page load requests /videos] --> AA[Get initial file list];
        AA --> BB[Initialize playlist];
        CC[Connects to /video_stream] --> DD[Receive filename via SSE];
        DD --> EE[Add filename to playlist];
        BB & EE --> FF[Select random video from playlist];
        FF --> GG[Request /videos/{filename}];
        GG --> HH[Receive video data & play];
    end

    FastAPI_Backend -- Manages --> File_System_Watcher;
```
