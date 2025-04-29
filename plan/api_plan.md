# プロジェクトAPI化計画

## 1. 目的

現在の Gradio インターフェースを介さずに、HTTPリクエストを通じてプログラムから動画生成機能を呼び出せるようにする。

## 2. 方針

* **APIフレームワーク:** **FastAPI** を採用します。
  * 理由: Python のモダンなWebフレームワークであり、非同期処理に強く、型ヒントによる自動ドキュメント生成機能があり、今回の要件に適しているため。
* **コード構成:** 既存の `demo_gradio.py` を分割し、APIサーバー、動画生成ワーカー、キュー管理などの役割ごとにモジュール化します。
  * `api.py`: FastAPI アプリケーション、APIエンドポイントの定義
  * `worker.py`: 動画生成処理 (`worker` 関数とその依存関係)
  * `queue_manager.py`: ジョブキューの管理 (クラス `QueuedJob`、`add_to_queue`, `get_next_job` など)
  * `models.py`: Hugging Face モデルの読み込み・管理
  * `settings.py`: 設定値 (ポート番号、モデルパスなど)
  * `requirements.txt`: 依存ライブラリ (FastAPIなどを追加)
* **APIエンドポイント:**
  * `POST /generate`: 動画生成ジョブをキューに追加する。
    * 入力: プロンプト、入力画像 (Base64エンコード文字列 or アップロードファイル)、動画長、シード値、各種パラメータ
    * 出力: ジョブID (即時応答)
  * `GET /status/{job_id}`: 指定したジョブのステータス (pending, processing, completed, failed) を確認する。
    * 入力: ジョブID
    * 出力: ジョブステータス
  * `GET /result/{job_id}`: 生成された動画ファイルを取得する。
    * 入力: ジョブID
    * 出力: 動画ファイル (MP4) or エラーメッセージ
  * `GET /queue`: 現在のジョブキューの状態を表示する (オプション)。
* **非同期処理:** 既存のジョブキュー (`job_queue.json` を利用する方式) を活用し、APIリクエストは即座に応答を返し、実際の動画生成はバックグラウンドで実行します。将来的には、より堅牢なタスクキュー (Celery など) への移行も検討可能です。
* **Gradioコードの分離:** Gradio 固有の UI コードは API 化には含めません。必要であれば、API を呼び出す別の Gradio アプリケーションとして再構築することも可能です。
* **利用シーン:** ローカルネットワーク上でのPoC (Proof of Concept)
* **認証:** 不要

## 3. 実装ステップ

1. **環境設定:** FastAPI と関連ライブラリをインストールし、`requirements.txt` を更新します。
2. **コード分割:** `demo_gradio.py` を上記のモジュール構成に分割・整理します。
3. **API実装:** `api.py` に FastAPI アプリケーションとエンドポイントを実装します。入力データのバリデーションも行います。
4. **ワーカー連携:** API が受け付けたリクエストを `queue_manager.py` を介してジョブキューに追加し、バックグラウンドで `worker.py` が処理を実行できるようにします。
5. **ステータス/結果取得:** `/status` および `/result` エンドポイントを実装し、ジョブの状態や生成結果を取得できるようにします。
6. **テスト:** 各エンドポイントが期待通りに動作するかテストします。
7. **ドキュメント:** FastAPI の自動ドキュメント機能を確認し、必要に応じて補足情報を追記します。

## 4. Mermaid ダイアグラム (簡易構成図)

```mermaid
graph LR
    Client[API Client] -- POST /generate --> API[FastAPI App (api.py)]
    API -- Add Job --> QM[Queue Manager (queue_manager.py)]
    QM -- Writes --> JobQueue[(job_queue.json)]
    Worker[Background Worker (worker.py)] -- Reads --> JobQueue
    Worker -- Processes Job --> Models[ML Models (models.py)]
    Worker -- Saves Result --> Outputs[(outputs/)]
    Client -- GET /status/{job_id} --> API
    API -- Get Status --> QM
    Client -- GET /result/{job_id} --> API
    API -- Reads Result --> Outputs
```
