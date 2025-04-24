# 進捗表示実装計画 (SSE利用)

## 目的

FastAPIバックエンドで実行される長時間ジョブの進捗状況を、React.jsフロントエンドにリアルタイムで表示する。

## 方針

Server-Sent Events (SSE) を利用して、サーバーからクライアントへ進捗更新をプッシュ通知する。

## 最終計画

1. **サーバー側 (FastAPI - `api/api.py`)**:
    * `/stream/status/{job_id}` エンドポイントを追加。
    * `StreamingResponse` を使用し、`Content-Type: text/event-stream` を設定。
    * エンドポイント内で非同期ループを開始。
    * ループ内で短い間隔（例: 0.5秒〜1秒）で `queue_manager.get_job_by_id(job_id)` を呼び出し、進捗状況を取得。
    * 取得した進捗状況が前回送信時から変化した場合、または一定時間ごとに、SSE形式 (`event: progress\ndata: JSON文字列\n\n`) でクライアントに進捗情報を送信。
    * ジョブステータスが "completed" または "failed" になったら、最終ステータスを送信 (`event: status`) し、ループを終了（ストリームが閉じる）。
2. **クライアント側 (React.js)**:
    * `useEffect` フック内で `EventSource` を使用して `/stream/status/{job_id}` に接続。
    * `EventSource` の `onmessage` または `addEventListener('progress', ...)` でサーバーからのイベントを受信。
    * 受信した `event.data` (JSON文字列) をパースし、Reactのstateを更新してUI（プログレスバー、ステータステキストなど）に反映。
    * `addEventListener('status', ...)` で最終ステータスを受信したら、`EventSource` を閉じる (`eventSource.close()`)。
    * コンポーネントのアンマウント時に `EventSource` を確実に閉じる処理を追加。

## Mermaid図 (処理フロー)

```mermaid
sequenceDiagram
    participant Client (React)
    participant API (FastAPI)
    participant QueueManager
    participant Worker

    Client->>+API: POST /generate (job details)
    API->>+QueueManager: add_to_queue(job details)
    QueueManager-->>-API: job_id
    API-->>-Client: {job_id: "..."}

    Client->>+API: GET /stream/status/{job_id} (SSE connection)
    API-->>-Client: (Connection established, Content-Type: text/event-stream)

    loop Monitor Job Progress (e.g., every 0.5-1s)
        API->>+QueueManager: get_job_by_id(job_id)
        QueueManager-->>-API: job_status (incl. progress)
        alt Progress Updated or Periodic Update
            API-->>Client: event: progress\ndata: {"status": "...", "progress": ..., ...}\n\n
        end
    end

    Worker->>+QueueManager: update_job_progress(job_id, ...)
    QueueManager->>QueueManager: Save queue file (job_queue.json)

    alt Job Completed/Failed
         API->>+QueueManager: get_job_by_id(job_id)
         QueueManager-->>-API: job_status (completed/failed)
         API-->>Client: event: status\ndata: {"status": "completed/failed", ...}\n\n
         Note over API, Client: Server closes stream, client closes EventSource
         Client->>-API: (Client closes EventSource)
    end

    Client->>+API: GET /result/{job_id}
    API->>+QueueManager: get_job_by_id(job_id)
    API->>API: Check output file exists
    API-->>-Client: Video File
```

## Reactクライアント実装例

以下は、指定された `jobId` の進捗をリアルタイムに表示するシンプルなReactコンポーネントの例です。

```jsx
import React, { useState, useEffect } from 'react';

function JobProgress({ jobId }) {
  const [status, setStatus] = useState('Connecting...');
  const [progress, setProgress] = useState(0);
  const [progressInfo, setProgressInfo] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId) {
      setStatus('No Job ID provided.');
      return;
    }

    setStatus('Connecting to stream...');
    setError(null); // Reset error on new jobId

    // Create EventSource connection
    const eventSource = new EventSource(`/stream/status/${jobId}`);

    // Handle 'progress' events
    eventSource.addEventListener('progress', (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus(data.status || 'Processing...');
        setProgress(data.progress || 0);
        setProgressInfo(data.progress_info || '');
        setError(null); // Clear error on successful data
      } catch (e) {
        console.error('Failed to parse progress data:', event.data, e);
        setError('Error parsing progress data.');
      }
    });

    // Handle 'status' events (for terminal states or errors)
    eventSource.addEventListener('status', (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus(data.status || 'Finished');
        setProgress(data.status === 'completed' ? 100 : progress); // Set 100% on completion
        setProgressInfo(data.message || 'Job finished.');
        setError(data.status === 'error' ? data.message : null);
        console.log(`Received final status for ${jobId}:`, data.status);
        eventSource.close(); // Close connection on final status
      } catch (e) {
        console.error('Failed to parse final status data:', event.data, e);
        setError('Error parsing final status data.');
        eventSource.close(); // Close on error too
      }
    });

    // Handle connection errors
    eventSource.onerror = (err) => {
      console.error('EventSource failed:', err);
      setStatus('Connection error');
      setError('Failed to connect to the status stream.');
      setProgress(0);
      setProgressInfo('');
      eventSource.close(); // Close connection on error
    };

    // Cleanup function to close connection when component unmounts or jobId changes
    return () => {
      console.log(`Closing EventSource for ${jobId}`);
      eventSource.close();
    };

  }, [jobId]); // Re-run effect if jobId changes

  return (
    <div>
      <h2>Job Progress (ID: {jobId || 'N/A'})</h2>
      <p>Status: {status}</p>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      <progress value={progress} max="100" style={{ width: '100%' }} />
      <p>{progress.toFixed(1)}%</p>
      {progressInfo && <p>Details: {progressInfo}</p>}
    </div>
  );
}

export default JobProgress;
```

**使い方:**

1. この `JobProgress` コンポーネントをReactアプリケーションに組み込みます。
2. 表示したいジョブのIDを `jobId` プロップとして渡します。

```jsx
import React from 'react';
import JobProgress from './JobProgress'; // Assuming the component is saved as JobProgress.js

function App() {
  const currentJobId = "your_job_id_here"; // Replace with the actual job ID

  return (
    <div className="App">
      <h1>Video Generation</h1>
      {/* Other components */}
      {currentJobId && <JobProgress jobId={currentJobId} />}
      {/* Other components */}
    </div>
  );
}

export default App;
