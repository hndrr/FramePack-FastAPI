import io
import os
import time
import threading
import traceback  # Added for exception printing
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
from typing import List

# Import modules created earlier (relative imports)
from . import settings
from . import models
from . import queue_manager
from . import worker

# --- FastAPI App Initialization ---
app = FastAPI(title="FramePack I2V API", version="0.1.0")

# --- Global State ---
# Dictionary to hold loaded models
loaded_models = {}
# Flag to indicate if the background worker is running
worker_running = False
worker_thread = None

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
    # Add more fields if needed, e.g., progress percentage, estimated time


class QueueStatusResponse(BaseModel):
    queue: List[dict]  # List of job summaries


# --- Background Worker ---
def background_worker_task():
    global worker_running
    print("Background worker started.")
    while worker_running:
        next_job = queue_manager.get_next_job()
        if next_job:
            print(f"Worker picked up job: {next_job.job_id}")
            try:
                # Ensure models are loaded before processing
                if not loaded_models:
                    print("Error: Models not loaded. Cannot process job.")
                    queue_manager.update_job_status(next_job.job_id, "failed - models not loaded")
                    continue  # Skip to next loop iteration

                worker.worker(next_job, loaded_models)
            except Exception as e:
                print(f"Unhandled exception in worker for job {next_job.job_id}: {e}")
                traceback.print_exc()
                try:
                    # Attempt to mark the job as failed even if worker crashed
                    queue_manager.update_job_status(next_job.job_id, f"failed - worker error: {type(e).__name__}")
                except Exception as update_e:
                    print(f"Critical: Failed to update job status after worker error: {update_e}")
            print(f"Worker finished processing job: {next_job.job_id}")
        else:
            # No job found, wait before checking again
            time.sleep(settings.WORKER_CHECK_INTERVAL)
    print("Background worker stopped.")

# --- FastAPI Events ---


@app.on_event("startup")
async def startup_event():
    global loaded_models, worker_running, worker_thread
    print("API starting up...")
    # Load models
    try:
        loaded_models = models.load_models(lora_path=settings.LORA_PATH)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load models on startup: {e}")
        # Depending on requirements, you might want to prevent startup
        # raise RuntimeError(f"Failed to load models: {e}") from e
        loaded_models = {}  # Ensure it's empty if loading failed

    # Start background worker
    if not worker_running:
        worker_running = True
        worker_thread = threading.Thread(target=background_worker_task, daemon=True)
        worker_thread.start()
        print("Background worker thread started.")


@app.on_event("shutdown")
def shutdown_event():
    global worker_running, worker_thread
    print("API shutting down...")
    # Stop background worker
    if worker_running:
        worker_running = False
        if worker_thread:
            print("Waiting for worker thread to finish...")
            worker_thread.join(timeout=settings.WORKER_CHECK_INTERVAL + 5)  # Wait a bit longer than check interval
            if worker_thread.is_alive():
                print("Warning: Worker thread did not stop gracefully.")
            else:
                print("Worker thread stopped.")
    # Cleanup resources if needed (e.g., explicitly delete models from GPU)
    # unload_models()  # Implement if necessary
    print("Shutdown complete.")


# --- API Endpoints ---

@app.post("/generate", response_model=GenerateResponse)
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    video_length: float = Form(5.0),
    seed: int = Form(-1),
    use_teacache: bool = Form(False),
    gpu_memory_preservation: float = Form(0.0),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    gs: float = Form(1.0),
    rs: float = Form(1.0),
    mp4_crf: float = Form(16.0),
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
    # Implementation needed: Fetch job status from queue_manager
    job = queue_manager.get_job_by_id(job_id)
    if not job:
        # Check if the job might have completed and its output exists
        output_file = os.path.join(settings.OUTPUTS_DIR, f"{job_id}.mp4")
        if os.path.exists(output_file):
            return JobStatusResponse(job_id=job_id, status="completed")
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(job_id=job.job_id, status=job.status)


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


# --- Main execution (for running with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    print(f"Starting Uvicorn server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)