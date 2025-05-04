# APIでのビデオ生成モード選択計画

`api/` ディレクトリ内のコードを変更し、`demo_gradio_f1.py` と同様の生成方法（順方向サンプリングと対応するモデル）も選択できるようにする計画。

**目標:** API 経由でビデオ生成をリクエストする際に、従来の逆方向サンプリング (`base`) と、`f1` スタイルの順方向サンプリング (`forward`) を選択可能にする。

**具体的な変更点:**

1. **モデルのロード (`api/models.py`):**
    * `load_models` 関数で、既存の `transformer` (`lllyasviel/FramePackI2V_HY`) に加えて、`f1` スタイルの `transformer_f1` (`lllyasviel/FramePack_F1_I2V_HY_20250503`) もロードするように変更します。
    * ロードされたモデルは、区別できるキー（例: `'transformer_base'`, `'transformer_f1'`）で辞書に格納します。

2. **ジョブキュー (`api/queue_manager.py`):**
    * `QueuedJob` データクラスに、以下のフィールドを追加します。
        * `sampling_mode: str` (値: `"reverse"` または `"forward"`, デフォルト: `"reverse"`)
        * `transformer_model: str` (値: `"base"` または `"f1"`, デフォルト: `"base"`)
    * `to_dict`, `from_dict` メソッドを更新し、新しいフィールドを含めます。
    * `add_to_queue` 関数の引数に `sampling_mode` と `transformer_model` を追加し、`QueuedJob` オブジェクト生成時にこれらの値を設定するようにします。

3. **API エンドポイント (`api/api.py`):**
    * `/generate` エンドポイントの `Form` パラメータに以下を追加します。
        * `sampling_mode: str = Form("reverse", description="Sampling loop direction ('reverse' or 'forward').")`
        * `transformer_model: str = Form("base", description="Transformer model to use ('base' or 'f1').")`
    * `queue_manager.add_to_queue` を呼び出す際に、これらの新しいパラメータを渡します。

4. **ワーカーロジック (`api/worker.py`):**
    * `worker` 関数の冒頭で、`job` オブジェクトから `sampling_mode` と `transformer_model` を取得します。
    * `transformer_model` の値に基づいて、`models` 辞書から適切な Transformer モデルを選択して使用します。
    * サンプリングループの部分を `if job.sampling_mode == "forward":` と `else:` で分岐させます。
        * **`forward` の場合:** `demo_gradio_f1.py` の L188-L287 のロジック（`history_latents` の初期化・更新、`sample_hunyuan` への引数準備、`vae_decode`, `soft_append_bcthw` の呼び出し）を実装します。
        * **`else` (`reverse`) の場合:** 現在の `api/worker.py` の L341-L575 のロジックを維持します。
    * `callback` 関数内のプログレス計算も、選択されたモードに応じて適切に表示されるように調整します（特にステップの進捗を示すテキスト）。

**処理フロー図 (Mermaid):**

```mermaid
graph TD
    A[API Request /generate] -- Job Params (prompt, image, sampling_mode, transformer_model...) --> B(api.py: generate_video);
    B -- image_np, params --> C(queue_manager.py: add_to_queue);
    C -- Creates QueuedJob (with mode/model) --> D(job_queue.json);
    E(background_worker_task) -- Checks queue --> F(queue_manager.py: get_next_job);
    F -- Reads job_queue.json --> G{Job Found?};
    G -- Yes --> H(Returns QueuedJob);
    G -- No --> E;
    H -- QueuedJob, loaded_models --> I(worker.py: worker);
    I -- Gets transformer_model --> J{Select Transformer};
    J -- transformer_model == 'f1' --> K[Use Transformer F1];
    J -- else --> L[Use Transformer Base];
    I -- Gets sampling_mode --> M{Select Sampling Loop};
    M -- sampling_mode == 'forward' --> N[Forward Sampling Loop (f1 style)];
    M -- else --> O[Reverse Sampling Loop (base style)];
    K & N -- Use F1 Model & Forward Loop --> P(Prepare Args for F1);
    L & O -- Use Base Model & Reverse Loop --> Q(Prepare Args for Base);
    P -- Calls sample_hunyuan --> R(Generate Latents);
    Q -- Calls sample_hunyuan --> R;
    R -- Latents --> S(VAE Decode & Append);
    S -- Pixels --> T(Save MP4);
    T -- MP4 Path --> U(Update Job Status: completed);
    I -- Updates Progress --> V(queue_manager.py: update_job_progress);
    I -- Updates Status --> W(queue_manager.py: update_job_status);

    subgraph Model Loading [api/models.py]
        direction LR
        ML1[Load Base Transformer]
        ML2[Load F1 Transformer]
        ML3[Load Other Models (VAE, Encoders...)]
    end

    subgraph Job Definition [api/queue_manager.py]
        direction LR
        JD1[QueuedJob Class]
        JD1 -- Add --> JD2[sampling_mode]
        JD1 -- Add --> JD3[transformer_model]
    end

    subgraph API Endpoint [api/api.py]
        direction LR
        AE1[/generate Endpoint]
        AE1 -- Add Form Param --> AE2[sampling_mode]
        AE1 -- Add Form Param --> AE3[transformer_model]
    end
```
