import io
import os
import time
import threading
import traceback
import asyncio
import json
import base64
import mimetypes
import logging
import enum
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from typing import List, Optional
from datetime import datetime
import uuid
from watchdog.observers import Observer

# Import modules created earlier (relative imports)
from . import settings
from . import models
from . import queue_manager
from . import worker
from . import video_watcher
from . import worker_image

# --- Global State ---
# Dictionary to hold loaded models
loaded_models = {}
# Flag to indicate if the background worker is running
worker_running = False
worker_thread = None
# Variable to store the ID of the currently processing job
currently_processing_job_id: str | None = None
# --- Video Watcher State ---
sse_clients: List[asyncio.Queue] = []  # List to hold client queues for SSE
observer: Observer | None = None  # type: ignore # Watchdog observer instance


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global loaded_models, worker_running, worker_thread, observer, sse_clients
    print("API starting up via lifespan...")
    # Load models (skip if in development mode)
    if os.getenv("MODELS_DISABLED", "false").lower() == "true":
        print("Models loading disabled - running in development mode")
        loaded_models = {}
    else:
        try:
            # Consider running blocking IO in a threadpool executor in async context
            # e.g., await asyncio.to_thread(models.load_models)
            # For simplicity now, keeping the direct call but be aware of potential blocking
            loaded_models = models.load_models()
            print("Models loaded successfully via lifespan.")
        except Exception as e:
            print(f"FATAL: Failed to load models on startup via lifespan: {e}")
            traceback.print_exc()
            loaded_models = {}

    # Start background worker
    if not worker_running:
        worker_running = True
        # Note: Starting/managing threads directly in async code needs care.
        worker_thread = threading.Thread(target=background_worker_task, daemon=True)
        worker_thread.start()
        print("Background worker thread started via lifespan.")
    else:
        print("Worker already running? Skipping start in lifespan.")

    # Start video watcher
    try:
        print(f"Attempting to start video watcher for directory: {settings.VIDEO_DIR}")
        # Pass the global sse_clients list to the watcher
        observer = video_watcher.start_watcher(settings.VIDEO_DIR, sse_clients)
        print("Video watcher started successfully via lifespan.")
    except Exception as e:
        print(f"FATAL: Failed to start video watcher on startup: {e}")
        traceback.print_exc()
        observer = None

    yield

    # Shutdown logic
    print("API shutting down via lifespan...")

    # Stop video watcher first
    if observer:
        try:
            print("Stopping video watcher...")
            video_watcher.stop_watcher(observer)
            print("Video watcher stopped.")
        except Exception as e:
            print(f"Error stopping video watcher: {e}")
            traceback.print_exc()

    # Stop background worker
    if worker_running:
        worker_running = False
        if worker_thread:
            print("Waiting for worker thread to finish via lifespan...")
            # Note: thread.join() is blocking. Consider alternatives in async context.
            worker_thread.join(timeout=settings.WORKER_CHECK_INTERVAL + 5)
            if worker_thread.is_alive():
                print("Warning: Worker thread did not stop gracefully via lifespan.")
            else:
                print("Worker thread stopped via lifespan.")
    # Cleanup resources
    print("Attempting to unload models...")
    try:
        models.unload_models(loaded_models)
        print("Models unloaded successfully (or placeholder executed).")
    except Exception as unload_e:
        print(f"Error during model unloading: {unload_e}")
        traceback.print_exc()
    print("Shutdown complete via lifespan.")


# --- FastAPI App Initialization ---
# Use the lifespan context manager
app = FastAPI(title="FramePack API", version="0.1.0", lifespan=lifespan)

# Set maximum file size (100MB)
app.state.max_request_size = 100 * 1024 * 1024

# --- CORS Middleware Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- End CORS Middleware Configuration ---


# --- Pydantic Models for API Requests/Responses ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation.")
    # image: str = Field(..., description="Base64 encoded input image.")
    video_length: float = Field(5.0, description="Length of the video in seconds.", gt=0)
    seed: int = Field(-1, description="Seed for generation. -1 for random.")
    use_teacache: bool = Field(False, description="Enable TEACache optimization.")
    gpu_memory_preservation: float = Field(10.0, description="GPU memory to preserve (GB) in low VRAM mode.", ge=0)
    steps: int = Field(20, description="Number of diffusion steps.", gt=0)
    cfg: float = Field(7.0, description="Classifier-Free Guidance scale.", ge=1.0)
    gs: float = Field(1.0, description="Guidance scale for start latent.", ge=0)
    rs: float = Field(1.0, description="Guidance scale for refinement.", ge=0)
    mp4_crf: float = Field(16.0, description="CRF value for MP4 encoding (lower means higher quality).", ge=0)
    # n_prompt: Optional[str] = Field("", description="Negative prompt.")  # Add if needed


class GenerateResponse(BaseModel):
    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    progress_step: Optional[int] = None
    progress_total: Optional[int] = None
    progress_info: Optional[str] = None


class QueueStatusResponse(BaseModel):
    queue: List[dict]  # List of job summaries


class WorkerStatusResponse(BaseModel):
    is_running: bool
    processing_job_id: Optional[str] = None


class LoraListResponse(BaseModel):
    loras: List[str]


class ResultResponse(BaseModel):
    video_url: str
    thumbnail_base64: Optional[str] = None


# --- Image Generation Models ---
class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for image generation.")
    negative_prompt: Optional[str] = Field("", description="Negative prompt.")
    input_image: str = Field(..., description="Base64 encoded input image (required).")
    seed: Optional[int] = Field(-1, description="Seed for generation. -1 for random.")
    steps: int = Field(30, description="Number of diffusion steps.", gt=0)
    cfg: float = Field(1.0, description="Classifier-Free Guidance scale.", ge=0.0)
    width: int = Field(1216, description="Image width.", gt=0)
    height: int = Field(704, description="Image height.", gt=0)
    gpu_memory_preservation: float = Field(10.0, description="GPU memory to preserve (GB) in low VRAM mode.", ge=0)
    lora_paths: Optional[List[str]] = Field(None, description="List of LoRA file paths.")
    lora_scales: Optional[List[float]] = Field(None, description="List of LoRA scales.")


class BatchImageRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of text prompts for batch generation.")
    input_image: str = Field(..., description="Base64 encoded input image (used for all batch items).")
    negative_prompt: Optional[str] = Field("", description="Negative prompt for all images.")
    seeds: Optional[List[int]] = Field(None, description="List of seeds (one per prompt).")
    batch_size: int = Field(4, description="Number of images to generate in parallel.", gt=0, le=8)
    steps: int = Field(30, description="Number of diffusion steps.", gt=0)
    cfg: float = Field(1.0, description="Classifier-Free Guidance scale.", ge=0.0)
    width: int = Field(1216, description="Image width.", gt=0)
    height: int = Field(704, description="Image height.", gt=0)
    gpu_memory_preservation: float = Field(10.0, description="GPU memory to preserve (GB) in low VRAM mode.", ge=0)
    lora_paths: Optional[List[str]] = Field(None, description="List of LoRA file paths.")
    lora_scales: Optional[List[float]] = Field(None, description="List of LoRA scales.")


class ImageTransferRequest(BaseModel):
    source_image: str = Field(..., description="Base64 encoded source image.")
    target_image: str = Field(..., description="Base64 encoded target image.")
    prompt: str = Field(..., description="Text prompt for the transfer.")
    negative_prompt: Optional[str] = Field("", description="Negative prompt.")
    transfer_strength: float = Field(0.7, description="Strength of the style transfer.", ge=0.0, le=1.0)
    seed: Optional[int] = Field(-1, description="Seed for generation. -1 for random.")
    steps: int = Field(30, description="Number of diffusion steps.", gt=0)
    cfg: float = Field(1.0, description="Classifier-Free Guidance scale.", ge=0.0)
    gpu_memory_preservation: float = Field(10.0, description="GPU memory to preserve (GB) in low VRAM mode.", ge=0)


class ImageGenerationResponse(BaseModel):
    job_id: str
    message: str




# --- Enum for Sampling Mode ---
class SamplingMode(str, enum.Enum):
    reverse = "reverse"
    forward = "forward"


# --- Enum for Output Type ---
class OutputType(str, enum.Enum):
    video = "video"
    image = "image"


# --- Background Worker ---
def background_worker_task():
    global worker_running, currently_processing_job_id
    print("Background worker started.")
    while worker_running:
        next_job = queue_manager.get_next_job()
        if next_job:
            currently_processing_job_id = next_job.job_id
            print(f"Worker picked up job: {currently_processing_job_id}")
            try:
                # Ensure models are loaded before processing
                if not loaded_models:
                    print("Error: Models not loaded. Cannot process job.")
                    queue_manager.update_job_status(currently_processing_job_id, "failed - models not loaded")
                    currently_processing_job_id = None
                    continue

                # Check job type and route to appropriate worker
                job_data = queue_manager.get_job_data_by_id(currently_processing_job_id)
                if job_data and isinstance(job_data, dict):
                    job_type = job_data.get("type", "video")
                    
                    if job_type == "image_generation":
                        worker_image.process_image_generation(currently_processing_job_id, job_data, loaded_models)
                    elif job_type == "batch_images":
                        worker_image.process_batch_images(currently_processing_job_id, job_data, loaded_models)
                    elif job_type == "image_transfer":
                        worker_image.process_image_transfer(currently_processing_job_id, job_data, loaded_models)
                    else:
                        # Default to video generation
                        worker.worker(next_job, loaded_models)
                else:
                    # Fallback to video generation for compatibility
                    worker.worker(next_job, loaded_models)
            except Exception as e:
                print(f"Unhandled exception in worker for job {currently_processing_job_id}: {e}")
                traceback.print_exc()
                try:
                    # Attempt to mark the job as failed even if worker crashed
                    queue_manager.update_job_status(currently_processing_job_id, f"failed - worker error: {type(e).__name__}")
                except Exception as update_e:
                    print(f"Critical: Failed to update job status after worker error: {update_e}")
            finally:
                # Ensure currently processing ID is cleared after job finishes (success or fail)
                print(f"Worker finished processing job: {currently_processing_job_id}")
                currently_processing_job_id = None
        else:
            # No job found, wait before checking again
            time.sleep(settings.WORKER_CHECK_INTERVAL)
    print("Background worker stopped.")

# --- API Endpoints ---

# === Job Execution Flow ===


@app.post("/generate", response_model=GenerateResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form("A character doing some simple body movements."),
    video_length: float = Form(5.0),
    seed: int = Form(-1),
    use_teacache: bool = Form(True),
    gpu_memory_preservation: float = Form(6.0),
    steps: int = Form(25),
    cfg: float = Form(1.0),
    gs: float = Form(10.0),
    rs: float = Form(0.0),
    mp4_crf: float = Form(16.0),
    lora_scale: float = Form(1.0),
    lora_path: Optional[str] = Form(None, description="Path to the LoRA file to use for this request (overrides server default if provided)."),
    sampling_mode: SamplingMode = Form(SamplingMode.reverse, description="Sampling loop direction."),
    image: UploadFile = File(...)
):
    """
    Accepts an image upload and text prompt to generate a video.
    Adds the job to the queue and returns the job ID immediately.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to load. API is not ready.")

    # Read and process the uploaded image
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        # Extract Exif data BEFORE converting to RGB or NumPy array
        original_exif = pil_image.info.get('exif')
        # Convert to RGB if necessary (e.g., if PNG has alpha)
        if pil_image.mode == 'RGBA':
            # Ensure Exif is preserved during conversion if possible (though convert might strip it)
            pil_image = pil_image.convert('RGB')
            # Re-check exif after convert? Might be lost.
        image_np = np.array(pil_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or process uploaded image: {e}")
    finally:
        await image.close()

    # Determine the transformer model based on sampling_mode
    # Use sampling_mode.value to get the string value from the Enum
    if sampling_mode == SamplingMode.forward:
        actual_transformer_model = "f1"
    elif sampling_mode == SamplingMode.reverse:
        actual_transformer_model = "base"
    else:
        # This 'else' block might be unreachable if using Enum correctly,
        # but kept for safety or future expansion. FastAPI handles invalid Enum values.
        print(f"Warning: Unexpected sampling_mode '{sampling_mode.value}'. Defaulting transformer_model to 'base'.")
        actual_transformer_model = "base"

    # Add job to the queue using queue_manager
    try:
        job_id = queue_manager.add_to_queue(
            prompt=prompt,
            image=image_np,
            original_exif=original_exif,
            video_length=video_length,
            seed=seed,
            use_teacache=use_teacache,
            gpu_memory_preservation=gpu_memory_preservation,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            mp4_crf=mp4_crf,
            lora_scale=lora_scale,
            lora_path=lora_path,
            sampling_mode=sampling_mode.value,
            transformer_model=actual_transformer_model,
            status="pending"
        )
    except Exception as e:
        print(f"Error adding job via queue_manager: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to add job to queue: {e}")

    if job_id is None:
        # This case might happen if save_image_to_temp failed inside add_to_queue
        raise HTTPException(status_code=500, detail="Failed to add job to queue (job ID is None). Check server logs.")

    print(f"Job added to queue with ID: {job_id}")
    return GenerateResponse(job_id=job_id, message="Video generation job added to queue.")


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Checks the status of a job."""
    global currently_processing_job_id

    # 1. Check if it's the currently processing job (and get its progress)
    if job_id == currently_processing_job_id:
        # Even if processing, get the latest progress details from the queue manager
        job_details = queue_manager.get_job_by_id(job_id)
        if job_details:
            return JobStatusResponse(
                job_id=job_id,
                status="processing",
                progress=getattr(job_details, 'progress', None),
                progress_step=getattr(job_details, 'progress_step', None),
                progress_total=getattr(job_details, 'progress_total', None),
                progress_info=getattr(job_details, 'progress_info', None)
            )
        else:
            # Should ideally not happen if it's the current job, but handle gracefully
            return JobStatusResponse(job_id=job_id, status="processing", progress_info="Details temporarily unavailable")

    # 2. Check if the job exists in the queue file (pending, failed, potentially completed but file not checked yet)
    job_in_file = queue_manager.get_job_by_id(job_id)
    if job_in_file:
        # Return the status and progress details from the file
        return JobStatusResponse(
            job_id=job_id,
            status=job_in_file.status,
            progress=getattr(job_in_file, 'progress', None),
            progress_step=getattr(job_in_file, 'progress_step', None),
            progress_total=getattr(job_in_file, 'progress_total', None),
            progress_info=getattr(job_in_file, 'progress_info', None)
        )

    # 3. Check if the output file exists (implies completed)
    output_file = os.path.join(settings.OUTPUTS_DIR, f"{job_id}.mp4")
    if os.path.exists(output_file):
        # If file exists, assume completed with 100% progress
        return JobStatusResponse(
            job_id=job_id,
            status="completed",
            progress=100.0,
            progress_info="Completed"
        )

    # 4. If none of the above, the job is not found
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/stream/status/{job_id}")
async def stream_job_status(job_id: str, request: Request):
    """
    Streams the status and progress of a job using Server-Sent Events (SSE).
    """
    async def event_generator():
        last_data_sent = None
        # terminal_statuses = {"completed", "cancelled"}

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                print(f"Client disconnected from job {job_id} stream.")
                break

            job = queue_manager.get_job_by_id(job_id)

            if not job:
                # Handle case where job might be cleaned up or never existed
                # Send a final message and close
                error_data = json.dumps({"status": "error", "message": "Job not found or cleaned up."})
                yield f"event: status\ndata: {error_data}\n\n"
                print(f"Job {job_id} not found for streaming, closing connection.")
                break

            # Prepare data payload
            current_data = {
                "job_id": job.job_id,
                "status": job.status,
                "progress": getattr(job, 'progress', 0.0),
                "progress_step": getattr(job, 'progress_step', 0),
                "progress_total": getattr(job, 'progress_total', 0),
                "progress_info": getattr(job, 'progress_info', '')
            }
            current_data_json = json.dumps(current_data)

            # Send data only if it has changed since last time
            if current_data_json != last_data_sent:
                yield f"event: progress\ndata: {current_data_json}\n\n"
                last_data_sent = current_data_json
                print(f"Sent progress update for job {job_id}: Status {job.status}, Progress {current_data['progress']:.1f}%")

            # Check for terminal status (completed, cancelled, failed)
            is_terminal = job.status == "completed" or job.status == "cancelled" or job.status.startswith("failed")
            if is_terminal:
                # Send final status event if it hasn't been sent already
                if current_data_json != last_data_sent:
                    yield f"event: progress\ndata: {current_data_json}\n\n"
                    last_data_sent = current_data_json
                    print(f"Sent final progress update for job {job_id}: Status {job.status}")

                # Send a dedicated 'status' event to signal completion/failure/cancellation
                final_status_data = json.dumps({"status": job.status, "message": "Job finished."})
                yield f"event: status\ndata: {final_status_data}\n\n"
                print(f"Job {job_id} reached terminal state: {job.status}. Closing stream.")
                break
            else:
                # Wait before checking again only if not terminal
                await asyncio.sleep(1)  # Check every 1 second
            if is_terminal:
                # Send final status event
                final_data = json.dumps({"status": job.status, "message": "Job finished."})
                yield f"event: status\ndata: {final_data}\n\n"
                print(f"Job {job_id} reached terminal state: {job.status}. Closing stream.")
                break

            # Wait before checking again
            await asyncio.sleep(1)  # Check every 1 second

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_job_result(job_id: str, request: Request):
    """
    Returns the download URL for the completed video and the Base64 encoded thumbnail.
    """
    job = queue_manager.get_job_by_id(job_id)
    output_file = os.path.join(settings.OUTPUTS_DIR, f"{job_id}.mp4")
    is_completed = (job and job.status == "completed") or (not job and os.path.exists(output_file))

    if not is_completed:
        if job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' is not completed yet (status: {job.status}).")
        else:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or result file does not exist.")

    # --- サムネイル処理 ---
    thumbnail_base64 = None
    if job and job.thumbnail and os.path.exists(job.thumbnail):
        try:
            with open(job.thumbnail, "rb") as f:
                thumbnail_data = f.read()
            thumbnail_base64_data = base64.b64encode(thumbnail_data).decode("utf-8")
            # MIMEタイプを推測 (例: image/jpeg)
            mime_type, _ = mimetypes.guess_type(job.thumbnail)
            if mime_type:
                thumbnail_base64 = f"data:{mime_type};base64,{thumbnail_base64_data}"
            else:
                # MIMEタイプが不明な場合はデフォルトを使用（またはエラー処理）
                thumbnail_base64 = f"data:image/jpeg;base64,{thumbnail_base64_data}"
            print(f"Job {job_id}: Encoded thumbnail from {job.thumbnail}")
        except Exception as e:
            print(f"Job {job_id}: Error reading or encoding thumbnail {job.thumbnail}: {e}")
            # サムネイルの読み込み/エンコードに失敗してもエラーにはしない

    # --- 動画URL構築 ---
    # request.url を使用して絶対URLまたは相対URLを構築
    video_url = str(request.url_for('download_video', job_id=job_id))

    return ResultResponse(
        video_url=video_url,
        thumbnail_base64=thumbnail_base64
    )


# --- Download Endpoints ---

@app.get("/download/video/{job_id}")
async def download_video(job_id: str):
    """Downloads the generated video file."""
    output_file = os.path.join(settings.OUTPUTS_DIR, f"{job_id}.mp4")
    if os.path.exists(output_file):
        return FileResponse(output_file, media_type="video/mp4", filename=f"{job_id}.mp4")
    else:
        # Optionally check job status again here if needed
        job = queue_manager.get_job_by_id(job_id)
        if job:
            raise HTTPException(status_code=404, detail=f"Video file for job '{job_id}' not found, status is '{job.status}'.")
        else:
            raise HTTPException(status_code=404, detail=f"Video file for job '{job_id}' not found.")


@app.get("/input_image/{job_id}")
async def get_input_image(job_id: str):
    """
    Returns the input JPEG image file associated with a job, potentially including Exif metadata.
    """
    job = queue_manager.get_job_by_id(job_id)
    filename_base = f"queue_image_{job_id}.jpg"
    input_image_path_in_temp = os.path.join(settings.TEMP_QUEUE_IMAGES_DIR, filename_base)

    if not job:
        # Check if the image file exists even if job is not in queue (e.g., after cleanup)
        if os.path.exists(input_image_path_in_temp):
            print(f"Job {job_id} not in queue, but input image file found. Serving file.")
            # Return JPEG file
            return FileResponse(input_image_path_in_temp, media_type="image/jpeg", filename=f"input_{job_id}.jpg")
        else:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found and input image file does not exist.")

    # Job exists, use the path from the job object (which should also be .jpg now)
    input_image_path_from_job = job.image_path
    if not input_image_path_from_job or not os.path.exists(input_image_path_from_job):
        # As a fallback, check the expected path in temp again, in case job object path is stale
        if os.path.exists(input_image_path_in_temp):
            print(f"Warning: Job {job_id} image path mismatch or file missing at '{input_image_path_from_job}', but found at '{input_image_path_in_temp}'. Serving found file.")
            return FileResponse(input_image_path_in_temp, media_type="image/jpeg", filename=f"input_{job_id}.jpg")
        else:
            raise HTTPException(status_code=404, detail=f"Input image file not found for job '{job_id}' at expected paths.")

    # Return JPEG file using path from job object
    return FileResponse(input_image_path_from_job, media_type="image/jpeg", filename=f"input_{job_id}.jpg")


@app.post("/cancel/{job_id}", status_code=200)
async def cancel_job(job_id: str):
    """Requests cancellation of a job."""
    # Check if the job exists (optional but good practice)
    job = queue_manager.get_job_by_id(job_id)
    output_file = os.path.join(settings.OUTPUTS_DIR, f"{job_id}.mp4")

    if not job and not os.path.exists(output_file):
        # If job not in queue and output doesn't exist, it's likely invalid
        raise HTTPException(status_code=404, detail="Job not found")

    if job and job.status == "completed":
        return {"message": "Job is already completed."}
    if not job and os.path.exists(output_file):
        return {"message": "Job is already completed (output file exists)."}

    # Update the job status to cancelled
    updated = queue_manager.update_job_status(job_id, "cancelled")

    if updated:
        print(f"Cancellation requested for job {job_id}")
        return {"message": f"Cancellation requested for job {job_id}."}
    else:
        # This might happen if the job completed between the check and the update,
        # or if get_job_by_id failed unexpectedly after the initial check.
        # Re-check status to provide a more accurate response.
        final_check_job = queue_manager.get_job_by_id(job_id)
        if final_check_job and final_check_job.status == "completed":
            return {"message": "Job completed before cancellation could be fully processed."}
        elif not final_check_job and os.path.exists(output_file):
            return {"message": "Job completed before cancellation could be fully processed (output file exists)."}
        else:
            # If still not found or status isn't completed, raise internal error
            print(f"Failed to update status to cancelled for job {job_id}, job might not exist anymore.")
            raise HTTPException(status_code=500, detail="Failed to request job cancellation. Job might have finished or encountered an issue.")


# === Queue & Worker Management ===

@app.get("/queue", response_model=QueueStatusResponse)
async def get_queue_info():
    # Implementation needed: Get queue status from queue_manager
    queue_status = queue_manager.get_queue_status()
    return QueueStatusResponse(queue=queue_status)


@app.get("/worker/status", response_model=WorkerStatusResponse)
async def get_worker_status():
    """Returns the current status of the background worker."""
    global worker_running, currently_processing_job_id
    return WorkerStatusResponse(
        is_running=worker_running,
        processing_job_id=currently_processing_job_id
    )


@app.post("/cleanup_jobs", status_code=200)
async def trigger_cleanup_jobs():
    """
    Manually triggers the cleanup of old completed, cancelled, or failed jobs
    based on the MAX_COMPLETED_JOBS setting.
    """
    try:
        removed_count = queue_manager.cleanup_jobs_by_max_count()
        return {"message": f"Cleanup process completed. Removed {removed_count} old job entries."}
    except Exception as e:
        print(f"Error during manual job cleanup: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to perform job cleanup: {e}")


# === Settings & Information ===

@app.get("/loras", response_model=LoraListResponse)
async def list_loras():
    """Lists available LoRA files from the configured directory."""
    lora_files = []
    allowed_extensions = {".safetensors", ".pt", ".bin"}
    try:
        if os.path.isdir(settings.LORA_DIR):
            for filename in os.listdir(settings.LORA_DIR):
                if os.path.isfile(os.path.join(settings.LORA_DIR, filename)):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in allowed_extensions:
                        lora_files.append(filename)
            lora_files.sort()
        else:
            print(f"Warning: LORA_DIR '{settings.LORA_DIR}' is not a valid directory.")
    except Exception as e:
        print(f"Error listing LoRA files: {e}")
        # Return empty list on error, or raise HTTPException
        # raise HTTPException(status_code=500, detail=f"Failed to list LoRA files: {e}")
    return LoraListResponse(loras=lora_files)


# === Video Streaming Endpoints ===

@app.get("/video_stream")
async def video_stream(request: Request):
    """
    Streams new video filenames using Server-Sent Events (SSE).
    """
    client_queue = asyncio.Queue()
    sse_clients.append(client_queue)
    logging.info(f"SSE client connected. Total clients: {len(sse_clients)}")

    async def event_generator():
        try:
            while True:
                # Check connection status first
                if await request.is_disconnected():
                    logging.info("SSE client disconnected.")
                    break

                try:
                    # Wait for a new filename from the queue
                    filename = await asyncio.wait_for(client_queue.get(), timeout=1.0)
                    logging.info(f"Sending SSE data: {filename}")
                    yield f"data: {filename}\n\n"
                    client_queue.task_done()
                except asyncio.TimeoutError:
                    # No new file, continue loop to check connection status
                    continue
                except Exception as e:
                    logging.error(f"Error in SSE generator: {e}")
                    # Optionally send an error event to the client
                    # yield f"event: error\ndata: {json.dumps({'message': 'Internal server error'})}\n\n"
                    break
        finally:
            # Cleanup when client disconnects or loop breaks
            if client_queue in sse_clients:
                sse_clients.remove(client_queue)
            logging.info(f"SSE client queue removed. Total clients: {len(sse_clients)}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/videos/{filename}")
async def get_video(filename: str):
    """
    Serves a specific video file from the VIDEO_DIR.
    """
    # Basic security check: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    filepath = os.path.join(settings.VIDEO_DIR, filename)
    logging.info(f"Request for video file: {filepath}")

    if not os.path.exists(filepath) or not os.path.isfile(filepath):
        logging.warning(f"Video file not found: {filepath}")
        raise HTTPException(status_code=404, detail="Video file not found")

    # Check if the file is an mp4 file (optional but recommended)
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid file type, only MP4 is supported.")

    return FileResponse(filepath, media_type="video/mp4", filename=filename)


@app.get("/videos", response_model=List[str])
async def list_videos():
    """
    Lists all .mp4 files currently in the VIDEO_DIR.
    """
    try:
        all_files = os.listdir(settings.VIDEO_DIR)
        mp4_files = sorted([f for f in all_files if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(settings.VIDEO_DIR, f))])
        logging.info(f"Found {len(mp4_files)} MP4 files in {settings.VIDEO_DIR}")
        return mp4_files
    except FileNotFoundError:
        logging.error(f"VIDEO_DIR not found: {settings.VIDEO_DIR}")
        raise HTTPException(status_code=500, detail="Video directory not found on server.")
    except Exception as e:
        logging.error(f"Error listing videos in {settings.VIDEO_DIR}: {e}")
        raise HTTPException(status_code=500, detail="Error listing video files.")


# --- Image Generation Endpoints ---

@app.post("/api/generate-image", response_model=ImageGenerationResponse)
async def generate_image(
    prompt: str = Form(..., description="Text prompt for image generation."),
    negative_prompt: str = Form("", description="Negative prompt."),
    seed: int = Form(-1, description="Seed for generation. -1 for random."),
    steps: int = Form(30, description="Number of diffusion steps."),
    cfg: float = Form(1.0, description="Classifier-Free Guidance scale."),
    width: int = Form(1216, description="Image width."),
    height: int = Form(704, description="Image height."),
    lora_path: Optional[str] = Form("", description="LoRA file path."),
    lora_scale: float = Form(1.0, description="LoRA scale."),
    gpu_memory_preservation: float = Form(10.0, description="GPU memory to preserve (GB) in low VRAM mode."),
    sampling_mode: str = Form("dpm-solver++", description="Sampling mode."),
    transformer_model: str = Form("base", description="Transformer model (base or f1)."),
    input_image: UploadFile = File(..., description="Input image file (required).", media_type="image/*")
):
    """
    Generate a single high-quality image from input image and text prompt (Image-to-Image).
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to load. API is not ready.")
    
    # Read and encode input image
    try:
        print(f"Debug: input_image type = {type(input_image)}")
        
        # Handle different types of input_image
        if isinstance(input_image, str):
            # If it's already a string, assume it's base64 encoded
            print("Debug: input_image is a string, treating as base64")
            input_image_b64 = input_image
        elif hasattr(input_image, 'read'):
            # It's an UploadFile
            print(f"Debug: input_image is UploadFile")
            if hasattr(input_image, 'filename'):
                print(f"Debug: filename = {input_image.filename}")
            if hasattr(input_image, 'content_type'):
                print(f"Debug: content_type = {input_image.content_type}")
            
            contents = await input_image.read()
            print(f"Debug: Read {len(contents)} bytes from UploadFile")
            
            import base64
            input_image_b64 = base64.b64encode(contents).decode('utf-8')
            print(f"Debug: Encoded to base64, length = {len(input_image_b64)}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid input_image type: {type(input_image)}. Expected UploadFile or base64 string.")
        
    except Exception as e:
        print(f"Debug: Exception occurred: {e}")
        print(f"Debug: Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error loading image: {e}")
    finally:
        if hasattr(input_image, 'close'):
            try:
                await input_image.close()
            except:
                pass
    
    # Create a unique job ID (8-digit hex like video generation)
    import uuid
    job_id = uuid.uuid4().hex[:8]
    
    # Prepare job data
    job_data = {
        "type": "image_generation",
        "data": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "input_image": input_image_b64,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "lora_path": lora_path,
            "lora_scale": lora_scale,
            "gpu_memory_preservation": gpu_memory_preservation,
            "sampling_mode": sampling_mode,
            "transformer_model": transformer_model
        },
        "created_at": datetime.now().isoformat()
    }
    
    # Add to queue
    try:
        queue_manager.add_job(job_id, job_data)
        print(f"Image generation job added to queue: {job_id}")
    except Exception as e:
        print(f"Error adding image job to queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add job to queue: {e}")
    
    # Jobs will be processed by the background worker
    
    return ImageGenerationResponse(
        job_id=job_id,
        message="Image generation job added to queue."
    )



@app.post("/api/batch-images", response_model=ImageGenerationResponse)
async def batch_images(request: BatchImageRequest):
    """
    Generate multiple images in batch from a list of prompts.
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to load. API is not ready.")
    
    # Create a unique job ID (8-digit hex like video generation)
    import uuid
    job_id = uuid.uuid4().hex[:8]
    
    # Prepare job data
    job_data = {
        "type": "batch_image",
        "data": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
    # Add to queue
    try:
        queue_manager.add_job(job_id, job_data)
        print(f"Batch image generation job added to queue: {job_id}")
    except Exception as e:
        print(f"Error adding batch image job to queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add job to queue: {e}")
    
    # Jobs will be processed by the background worker
    
    return ImageGenerationResponse(
        job_id=job_id,
        message="Batch image generation job added to queue."
    )


@app.post("/api/transfer-image", response_model=ImageGenerationResponse)
async def transfer_image(request: ImageTransferRequest):
    """
    Transfer style from source image to target image (kisekaeichi functionality).
    """
    if not loaded_models:
        raise HTTPException(status_code=503, detail="Models are not loaded or failed to load. API is not ready.")
    
    # Create a unique job ID (8-digit hex like video generation)
    import uuid
    job_id = uuid.uuid4().hex[:8]
    
    # Prepare job data
    job_data = {
        "type": "image_transfer",
        "data": request.dict(),
        "created_at": datetime.now().isoformat()
    }
    
    # Add to queue
    try:
        queue_manager.add_job(job_id, job_data)
        print(f"Image transfer job added to queue: {job_id}")
    except Exception as e:
        print(f"Error adding image transfer job to queue: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add job to queue: {e}")
    
    # Jobs will be processed by the background worker
    
    return ImageGenerationResponse(
        job_id=job_id,
        message="Image transfer job added to queue."
    )


@app.get("/api/image/{job_id}")
async def get_generated_image(job_id: str):
    """
    Download the generated image file.
    """
    # Check job status first
    job = queue_manager.get_job_by_id(job_id)
    if job and job.status != "completed":
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' is not completed yet (status: {job.status}).")
    
    # Check for image file
    image_path = os.path.join("outputs/images", f"{job_id}.png")
    if os.path.exists(image_path):
        return FileResponse(image_path, media_type="image/png", filename=f"{job_id}.png")
    else:
        raise HTTPException(status_code=404, detail=f"Image file for job '{job_id}' not found.")


# --- Main execution (for running with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    # Configure logging for the main execution context as well
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Starting Uvicorn server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run("api.api:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
