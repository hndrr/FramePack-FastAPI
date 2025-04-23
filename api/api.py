import io
import os
import time
import threading
import traceback
from contextlib import asynccontextmanager  # Import from standard library
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from typing import List, Optional  # Import Optional (Dict removed as unused)

# Import modules created earlier (relative imports)
from . import settings
from . import models
from . import queue_manager
from . import worker

# --- Global State ---
# Dictionary to hold loaded models
loaded_models = {}
# Flag to indicate if the background worker is running
worker_running = False
worker_thread = None
# Variable to store the ID of the currently processing job
currently_processing_job_id: str | None = None


# --- Lifespan Context Manager ---
@asynccontextmanager  # Use the imported decorator directly
async def lifespan(app: FastAPI):
    # Startup logic
    global loaded_models, worker_running, worker_thread
    print("API starting up via lifespan...")
    # Load models
    try:
        # Consider running blocking IO in a threadpool executor in async context
        # e.g., await asyncio.to_thread(models.load_models)  # lora_path removed
        # For simplicity now, keeping the direct call but be aware of potential blocking
        loaded_models = models.load_models()  # lora_path 引数を削除
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

    yield

    # Shutdown logic
    print("API shutting down via lifespan...")
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
        models.unload_models(loaded_models)  # Call the function from models module
        print("Models unloaded successfully (or placeholder executed).")
    except Exception as unload_e:
        print(f"Error during model unloading: {unload_e}")
        traceback.print_exc()
    print("Shutdown complete via lifespan.")


# --- FastAPI App Initialization ---
# Use the lifespan context manager
app = FastAPI(title="FramePack API", version="0.1.0", lifespan=lifespan)


# --- Pydantic Models for API Requests/Responses ---
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation.")
    # image: str = Field(..., description="Base64 encoded input image.")  # Changed to use UploadFile
    video_length: float = Field(5.0, description="Length of the video in seconds.", gt=0)
    seed: int = Field(-1, description="Seed for generation. -1 for random.")
    use_teacache: bool = Field(False, description="Enable TEACache optimization.")
    gpu_memory_preservation: float = Field(0.0, description="GPU memory to preserve (GB) in low VRAM mode.", ge=0)
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


# --- Background Worker ---
def background_worker_task():
    global worker_running, currently_processing_job_id
    print("Background worker started.")
    while worker_running:
        next_job = queue_manager.get_next_job()
        if next_job:
            currently_processing_job_id = next_job.job_id  # Set current job ID
            print(f"Worker picked up job: {currently_processing_job_id}")
            try:
                # Ensure models are loaded before processing
                if not loaded_models:
                    print("Error: Models not loaded. Cannot process job.")
                    queue_manager.update_job_status(currently_processing_job_id, "failed - models not loaded")
                    currently_processing_job_id = None  # Clear current job ID on error
                    continue  # Skip to next loop iteration

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


@app.post("/generate", response_model=GenerateResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form("A character doing some simple body movements."),  # Set default prompt
    video_length: float = Form(5.0),
    seed: int = Form(-1),
    use_teacache: bool = Form(True),  # Default to True (matching demo_gradio.py)
    gpu_memory_preservation: float = Form(6.0),  # Default to 6.0 GB (matching demo_gradio.py)
    steps: int = Form(25),
    cfg: float = Form(1.0),
    gs: float = Form(10.0),
    rs: float = Form(0.0),
    mp4_crf: float = Form(16.0),
    lora_scale: float = Form(1.0),  # 追加: LoRA強度パラメータ
    lora_path: Optional[str] = Form(None, description="Path to the LoRA file to use for this request (overrides server default if provided)."),  # 追加: LoRAファイルパス
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
        # Convert to RGB if necessary (e.g., if PNG has alpha)
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        image_np = np.array(pil_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read or process uploaded image: {e}")
    finally:
        await image.close()

    # Add job to the queue using queue_manager
    try:
        job_id = queue_manager.add_to_queue(
            prompt=prompt,
            image=image_np,  # Pass the numpy array
            video_length=video_length,
            seed=seed,
            use_teacache=use_teacache,
            gpu_memory_preservation=gpu_memory_preservation,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            mp4_crf=mp4_crf,
            lora_scale=lora_scale,  # 追加: lora_scale を渡す
            lora_path=lora_path,    # 追加: lora_path を渡す
            status="pending"  # Explicitly set initial status
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

# --- Placeholder Endpoints (to be implemented next) ---


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
    job_in_file = queue_manager.get_job_by_id(job_id)  # Use the function that reads file
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


@app.get("/result/{job_id}")
async def get_job_result(job_id: str):
    # Implementation needed: Check job status, return video file if completed
    job = queue_manager.get_job_by_id(job_id)
    output_file = os.path.join(settings.OUTPUTS_DIR, f"{job_id}.mp4")

    if job and job.status == "completed" and os.path.exists(output_file):
        return FileResponse(output_file, media_type="video/mp4", filename=f"{job_id}.mp4")
    elif not job and os.path.exists(output_file):
        # If job not in queue but file exists, assume completed
        print(f"Job {job_id} not in queue, but result file found. Serving file.")
        return FileResponse(output_file, media_type="video/mp4", filename=f"{job_id}.mp4")
    elif job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' is not completed yet (status: {job.status}).")
    else:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or result file does not exist.")


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


@app.get("/loras", response_model=LoraListResponse)
async def list_loras():
    """Lists available LoRA files from the configured directory."""
    lora_files = []
    allowed_extensions = {".safetensors", ".pt", ".bin"}  # Common LoRA extensions
    try:
        if os.path.isdir(settings.LORA_DIR):
            for filename in os.listdir(settings.LORA_DIR):
                if os.path.isfile(os.path.join(settings.LORA_DIR, filename)):
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in allowed_extensions:
                        lora_files.append(filename)
            lora_files.sort()  # Sort alphabetically
        else:
            print(f"Warning: LORA_DIR '{settings.LORA_DIR}' is not a valid directory.")
    except Exception as e:
        print(f"Error listing LoRA files: {e}")
        # Return empty list on error, or raise HTTPException
        # raise HTTPException(status_code=500, detail=f"Failed to list LoRA files: {e}")
    return LoraListResponse(loras=lora_files)


# --- Main execution (for running with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    print(f"Starting Uvicorn server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
