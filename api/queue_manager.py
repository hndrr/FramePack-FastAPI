import os
import json
import traceback
import uuid
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from PIL import Image
# from PIL.PngImagePlugin import PngInfo # No longer needed for JPEG saving

from . import settings  # Import settings to get paths


# Queue file path (from settings)
QUEUE_FILE = settings.QUEUE_FILE_PATH

# Temp directory for queue images (from settings)
temp_queue_images = settings.TEMP_QUEUE_IMAGES_DIR
# os.makedirs(temp_queue_images, exist_ok=True) # Directory creation handled in settings.py


@dataclass
class QueuedJob:
    prompt: str
    image_path: str
    video_length: float
    job_id: str  # Changed to string for hex ID
    seed: int
    use_teacache: bool
    gpu_memory_preservation: float
    steps: int
    cfg: float
    gs: float
    rs: float
    status: str = "pending"
    thumbnail: str = ""
    mp4_crf: float = 16
    # Progress tracking fields
    progress: float = 0.0
    progress_step: int = 0
    progress_total: int = 0  # Default to 0, will be set by worker
    progress_info: str = ""
    lora_scale: float = 1.0  # 追加: LoRA強度
    lora_path: Optional[str] = None
    # Add sampling mode and transformer model selection
    sampling_mode: str = "reverse"  # "reverse" or "forward"
    transformer_model: str = "base"  # "base" or "f1"
    # Add updated_at timestamp, default to current UTC time
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Add field for original Exif data (bytes) - will not be saved in JSON
    original_exif: Optional[bytes] = field(default=None, repr=False)
    # Add job type field
    job_type: str = "video"  # "video", "image", "batch_image", "image_transfer"
    # Add generic data field for different job types
    data: Optional[Dict[str, Any]] = None
    # Add result field for completed jobs
    result: Optional[Dict[str, Any]] = None

    def to_dict(self):
        # Exclude original_exif from the dictionary saved to JSON
        try:
            # Convert datetime to ISO 8601 string format for JSON serialization
            updated_at_iso = self.updated_at.isoformat() if self.updated_at else None
            return {
                'prompt': self.prompt,
                'image_path': self.image_path,
                'video_length': self.video_length,
                'job_id': self.job_id,
                'seed': self.seed,
                'use_teacache': self.use_teacache,
                'gpu_memory_preservation': self.gpu_memory_preservation,
                'steps': self.steps,
                'cfg': self.cfg,
                'gs': self.gs,
                'rs': self.rs,
                'status': self.status,
                'thumbnail': self.thumbnail,
                'mp4_crf': self.mp4_crf,
                # Progress fields
                'progress': self.progress,
                'progress_step': self.progress_step,
                'progress_total': self.progress_total,
                'progress_info': self.progress_info,
                'lora_scale': self.lora_scale,  # 追加
                'lora_path': self.lora_path,
                'sampling_mode': self.sampling_mode,  # Add sampling_mode
                'transformer_model': self.transformer_model,  # Add transformer_model
                'updated_at': updated_at_iso,  # Add updated_at
                'job_type': self.job_type,  # Add job_type
                'data': self.data,  # Add data
                'result': self.result,  # Add result
            }
        except Exception as e:
            print(f"Error converting job to dict: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        try:
            # Convert ISO 8601 string back to datetime object
            updated_at_iso = data.get('updated_at')
            updated_at_dt = None
            if updated_at_iso:
                try:
                    # Handle potential 'Z' suffix for UTC
                    if updated_at_iso.endswith('Z'):
                        updated_at_iso = updated_at_iso[:-1] + '+00:00'
                    updated_at_dt = datetime.fromisoformat(updated_at_iso)
                    # Ensure timezone-aware (assume UTC if no timezone info)
                    if updated_at_dt.tzinfo is None:
                        updated_at_dt = updated_at_dt.replace(tzinfo=timezone.utc)  # Corrected indentation and comment space
                except ValueError:
                    print(f"Warning: Could not parse updated_at timestamp '{updated_at_iso}'. Using current time.")
                    updated_at_dt = datetime.now(timezone.utc)
            else:
                # If updated_at is missing, default to now
                updated_at_dt = datetime.now(timezone.utc)

            return cls(
                prompt=data.get('prompt', 'A character doing some simple body movements.'),
                image_path=data.get('image_path', ''),
                video_length=data.get('video_length', 5.0),
                job_id=data.get('job_id', uuid.uuid4().hex[:8]),  # Provide default if missing
                seed=data.get('seed', -1),
                use_teacache=data.get('use_teacache', True),  # Default to True
                gpu_memory_preservation=data.get('gpu_memory_preservation', 0.0),
                steps=data.get('steps', 20),
                cfg=data.get('cfg', 7.0),
                gs=data.get('gs', 1.0),
                rs=data.get('rs', 1.0),
                status=data.get('status', 'pending'),
                thumbnail=data.get('thumbnail', ''),
                mp4_crf=data.get('mp4_crf', 16.0),
                # Progress fields with defaults
                progress=data.get('progress', 0.0),
                progress_step=data.get('progress_step', 0),
                progress_total=data.get('progress_total', 0),
                progress_info=data.get('progress_info', ''),
                lora_scale=data.get('lora_scale', 1.0),   # 追加
                lora_path=data.get('lora_path', None),
                sampling_mode=data.get('sampling_mode', 'reverse'),  # Add sampling_mode with default
                transformer_model=data.get('transformer_model', 'base'),  # Add transformer_model with default
                updated_at=updated_at_dt,  # Add updated_at
                job_type=data.get('job_type', 'video'),  # Add job_type with default
                data=data.get('data', None),  # Add data
                result=data.get('result', None),  # Add result
            )
        except Exception as e:
            print(f"Error creating job from dict: {str(e)}")
            traceback.print_exc()
            return None


# Initialize job queue as a list
job_queue = []

# Dictionary to hold the latest preview image (Base64) for processing jobs (in-memory only)
current_previews: Dict[str, str] = {}


def save_queue():
    global job_queue
    try:
        jobs = []
        for job in job_queue:
            job_dict = job.to_dict()
            if job_dict is not None:
                jobs.append(job_dict)

        file_path = os.path.abspath(QUEUE_FILE)
        with open(file_path, 'w') as f:
            json.dump(jobs, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving queue: {str(e)}")
        traceback.print_exc()
        return False


def load_queue_from_file() -> list[QueuedJob]:
    """Loads the queue from the JSON file and returns it as a list."""
    try:
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'r') as f:
                jobs_data = json.load(f)
            loaded_queue = []
            for job_data in jobs_data:
                job = QueuedJob.from_dict(job_data)
                if job is not None:
                    loaded_queue.append(job)
            return loaded_queue
        return []
    except Exception as e:
        print(f"Error loading queue: {str(e)}")
        traceback.print_exc()
        return []


# Load existing queue on startup into the global variable
job_queue = load_queue_from_file()

# Global storage for job data (for image generation jobs)
job_data_storage: Dict[str, dict] = {}


def save_image_to_temp(image: np.ndarray, job_id: str, prompt: str, seed: int, exif_data: Optional[bytes] = None) -> str:
    """Save image to temp directory as JPEG with Exif metadata and return the path"""
    try:
        # Convert numpy array to PIL Image
        squeezed_image = np.squeeze(image)
        pil_image = Image.fromarray(squeezed_image)
        # logging.info(f"[Job {job_id}] Exif in pil_image after fromarray: {pil_image.info.get('exif') is not None}") # DEBUG: Removed

        # Create unique filename using hex ID, change extension to jpg
        filename = f"queue_image_{job_id}.jpg"
        filepath = os.path.join(temp_queue_images, filename)

        # Prepare save arguments
        save_kwargs = {
            "format": "JPEG",
            "quality": 70,  # Lower quality for smaller file size
        }
        if exif_data:
            save_kwargs["exif"] = exif_data
            logging.info(f"[Job {job_id}] Attempting to save with Exif data.")
        else:
            logging.info(f"[Job {job_id}] No Exif data provided for saving.")

        # Save image as JPEG with or without Exif
        pil_image.save(filepath, **save_kwargs)
        # logging.info(f"[Job {job_id}] Saved temp image to {filepath} (JPEG)") # DEBUG: Removed
        return filepath
    except Exception as e:
        print(f"Error saving image with metadata: {str(e)}")
        traceback.print_exc()
        return ""


def add_to_queue(prompt, image, original_exif: Optional[bytes], video_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, status="pending", mp4_crf=16, lora_scale: float = 1.0, lora_path: Optional[str] = None, sampling_mode: str = "reverse", transformer_model: str = "base"):
    global job_queue
    try:
        # Generate a unique hex ID for the job
        job_id = uuid.uuid4().hex[:8]
        # Save image to temp directory and get path
        image_array = np.array(image)
        # Pass original_exif to save_image_to_temp
        image_path = save_image_to_temp(image_array, job_id, prompt, seed, exif_data=original_exif)
        if not image_path:
            print("Failed to save image with Exif to temp, cannot add job.")
            return None

        job = QueuedJob(
            prompt=prompt,
            image_path=image_path,
            video_length=video_length,
            job_id=job_id,
            seed=seed,
            use_teacache=use_teacache,
            gpu_memory_preservation=gpu_memory_preservation,
            steps=steps,
            cfg=cfg,
            gs=gs,
            rs=rs,
            status=status,
            mp4_crf=mp4_crf,
            lora_scale=lora_scale,
            lora_path=lora_path,
            sampling_mode=sampling_mode,  # Add sampling_mode
            transformer_model=transformer_model,  # Add transformer_model
            original_exif=original_exif  # Store exif in job object (won't be saved to JSON)
        )
        job_queue.append(job)
        if not save_queue():  # Save immediately after adding
            print("Failed to save queue after adding job.")
            # Optionally remove the job if saving fails, or handle differently
            # job_queue.pop()
            # return None
        return job_id
    except Exception as e:
        print(f"Error adding job to queue: {str(e)}")
        traceback.print_exc()
        return None


def add_job(job_id: str, job_data: Dict[str, Any]):
    """Add a generic job to the queue"""
    global job_queue, job_data_storage
    try:
        # Store job data for image generation jobs
        job_data_storage[job_id] = job_data
        
        job = QueuedJob(
            prompt=job_data.get("data", {}).get("prompt", ""),
            image_path="",  # Will be filled for video jobs
            video_length=0.0,  # Will be filled for video jobs
            job_id=job_id,
            seed=job_data.get("data", {}).get("seed", -1),
            use_teacache=job_data.get("data", {}).get("use_teacache", False),
            gpu_memory_preservation=job_data.get("data", {}).get("gpu_memory_preservation", 0.0),
            steps=job_data.get("data", {}).get("steps", 30),
            cfg=job_data.get("data", {}).get("cfg", 1.0),
            gs=job_data.get("data", {}).get("gs", 1.0),
            rs=job_data.get("data", {}).get("rs", 1.0),
            status="pending",
            job_type=job_data.get("type", "image"),
            data=job_data.get("data", {}),
            updated_at=datetime.now(timezone.utc)
        )
        job_queue.append(job)
        if not save_queue():
            print(f"Failed to save queue after adding job {job_id}")
            job_queue.pop()
            return False
        return True
    except Exception as e:
        print(f"Error adding job {job_id} to queue: {str(e)}")
        traceback.print_exc()
        return False


def update_job_result(job_id: str, result: Dict[str, Any]):
    """Update the result field of a job"""
    global job_queue
    job_updated = False
    
    # Try to update in memory first
    for job in job_queue:
        if job.job_id == job_id:
            job.result = result
            job.updated_at = datetime.now(timezone.utc)
            job_updated = True
            break
    
    if job_updated:
        save_queue()
    else:
        # Try loading from file and updating
        current_queue = load_queue_from_file()
        for job in current_queue:
            if job.job_id == job_id:
                job.result = result
                job.updated_at = datetime.now(timezone.utc)
                job_updated = True
                break
        
        if job_updated:
            try:
                jobs_to_save = [j.to_dict() for j in current_queue if j.to_dict() is not None]
                with open(QUEUE_FILE, 'w') as f:
                    json.dump(jobs_to_save, f, indent=2)
                job_queue = current_queue
            except Exception as e:
                print(f"Error saving queue after updating result for job {job_id}: {e}")
                traceback.print_exc()
                job_updated = False
    
    return job_updated


# Removed redundant traceback import


def get_next_job():
    """
    Gets the next pending job from the global in-memory queue,
    updates its status to 'processing' in the file, and returns the job object.
    Does NOT remove the job from the file until finalized.
    """
    global job_queue
    try:
        # Find the first 'pending' job in the in-memory queue
        job_to_process = None
        job_index = -1
        # Iterate through a copy of the queue indices to allow safe removal
        indices = list(range(len(job_queue)))
        for i in indices:
            # Check if index is still valid after potential removals by other threads/processes (unlikely here but safer)
            if i < len(job_queue):
                job = job_queue[i]
                if job.status == 'pending':
                    job_to_process = job
                    job_index = i
                    break  # Found the first pending job

        if job_to_process:
            # Remove from in-memory queue to prevent other workers taking it
            # Use del with index for potentially better performance than pop(index) in some scenarios
            del job_queue[job_index]
            logging.info(f"Worker picked up job {job_to_process.job_id}. Removed from in-memory pending list.")

            # Update status to 'processing' in the persistent queue file
            # We need to reload the queue from file, update the specific job, and save again.
            # This ensures we are working with the most current state from the file.
            current_persistent_queue = load_queue_from_file()
            job_found_in_file = False
            for job_in_file in current_persistent_queue:
                if job_in_file.job_id == job_to_process.job_id:
                    # Only update if it's still pending or somehow back to pending in the file
                    # If it's already processing/completed/failed by another interaction, log it.
                    if job_in_file.status == 'pending':
                        job_in_file.status = 'processing'
                        job_found_in_file = True
                        logging.info(f"Updating status to 'processing' for job {job_to_process.job_id} in the queue file.")
                    else:
                        # This might indicate a race condition or unexpected state.
                        logging.warning(f"Job {job_to_process.job_id} found in file but status is already '{job_in_file.status}', not 'pending'. Proceeding cautiously.")
                        # We still removed it from memory, so let the worker process it,
                        # but the file state might be inconsistent. Mark as found.
                        job_found_in_file = True  # Treat as found for saving logic below
                    break

            if job_found_in_file:
                # Save the updated queue back to the file
                try:
                    jobs_to_save = [j.to_dict() for j in current_persistent_queue if j.to_dict() is not None]
                    file_path = os.path.abspath(QUEUE_FILE)
                    with open(file_path, 'w') as f:
                        json.dump(jobs_to_save, f, indent=2)
                    logging.info(f"Queue file saved with job {job_to_process.job_id} marked as 'processing' (or existing status).")
                    # Return the job object that the worker will process
                    # Ensure the returned object also has the 'processing' status for the worker
                    job_to_process.status = 'processing'  # Set status for the object being returned
                    return job_to_process
                except Exception as e:
                    logging.error(f"Error saving queue after marking job {job_to_process.job_id} as processing: {e}")
                    traceback.print_exc()
                    # If saving fails, put the job back into the in-memory queue at the beginning
                    job_queue.insert(0, job_to_process)  # Re-insert at the beginning
                    logging.info(f"Re-inserted job {job_to_process.job_id} into memory queue due to save failure.")
                    return None
            else:
                # This case should ideally not happen if the job was just in the in-memory queue
                # loaded from the file, but handle defensively.
                logging.error(f"Job {job_to_process.job_id} was popped from memory but not found in the file for status update. This indicates a potential state inconsistency.")
                # Do not re-insert into memory queue as the file state is unknown/inconsistent.
                return None
        else:
            # No pending jobs found in the in-memory queue
            # logging.info("No pending jobs found in the in-memory queue.") # Optional: reduce log noise
            return None
    except Exception as e:
        logging.error(f"Error getting next job: {str(e)}")
        traceback.print_exc()
        return None


def get_job_data_by_id(job_id: str) -> Optional[dict]:
    """Get job data dictionary by job ID"""
    global job_data_storage
    return job_data_storage.get(job_id)


def get_job_by_id(job_id: str) -> QueuedJob | None:
    """Finds a job by its ID by reading the queue file directly."""
    logging.info(f"Attempting to get job by ID: {job_id}")
    current_queue = load_queue_from_file()  # Always read from file for this check
    for job in current_queue:
        if job.job_id == job_id:
            logging.info(f"Job {job_id} found. Progress: {job.progress}, Info: '{job.progress_info}'")
            return job
    logging.warning(f"Job {job_id} not found in queue file.")
    return None


def update_job_status(job_id: str, status: str, thumbnail: str = None):
    """Updates the status (and optionally thumbnail) of a job in the global queue and saves the file."""
    global job_queue
    job_updated = False  # Changed initial value to False
    job_found_in_memory = False
    for job in job_queue:
        if job.job_id == job_id:
            # Update status and timestamp
            if job.status != status:  # Only update timestamp if status actually changes
                job.status = status
                job.updated_at = datetime.now(timezone.utc)
                job_updated = True
            if thumbnail and job.thumbnail != thumbnail:
                job.thumbnail = thumbnail
                job.updated_at = datetime.now(timezone.utc)  # Also update if thumbnail changes
                job_updated = True  # Mark as updated if thumbnail changed

            if job_updated:
                job_found_in_memory = True
            break

    if job_updated:
        save_queue()  # Save if updated in memory

    # If not found or updated in memory, try loading from file, updating, and saving
    if not job_found_in_memory:
        current_queue = load_queue_from_file()
        job_found_in_file = False
        for job in current_queue:
            if job.job_id == job_id:
                # Update status and timestamp in file data
                if job.status != status:
                    job.status = status
                    job.updated_at = datetime.now(timezone.utc)
                    job_found_in_file = True
                if thumbnail and job.thumbnail != thumbnail:
                    job.thumbnail = thumbnail
                    job.updated_at = datetime.now(timezone.utc)
                    job_found_in_file = True  # Mark as found if thumbnail changed

                if job_found_in_file:
                    break

        if job_found_in_file:
            # Overwrite the file with the updated list
            try:
                jobs_to_save = [j.to_dict() for j in current_queue if j.to_dict() is not None]
                file_path = os.path.abspath(QUEUE_FILE)
                with open(file_path, 'w') as f:
                    json.dump(jobs_to_save, f, indent=2)
                job_updated = True  # Mark as updated since we saved the file
                # Update the global in-memory queue as well
                job_queue = current_queue  # Update global variable after successful save
            except Exception as e:
                print(f"Error saving queue after updating job {job_id} found in file: {e}")
                traceback.print_exc()
                job_updated = False  # Ensure update status reflects save failure
        else:
            print(f"Job with ID {job_id} not found in memory or file for status update.")

    # Clear preview if the job reached a terminal state
    is_terminal = status == "completed" or status == "cancelled" or status.startswith("failed")
    if is_terminal:
        clear_current_preview(job_id)

    return job_updated


# Configure logging (moved import to top)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Preview Data Management (In-Memory) ---

def update_current_preview(job_id: str, preview_base64: str):
    """Stores the latest preview image Base64 string for a job."""
    global current_previews
    current_previews[job_id] = preview_base64
    # logging.debug(f"Updated preview for job {job_id}") # Optional: Debug logging


def get_current_preview(job_id: str) -> Optional[str]:
    """Retrieves the latest preview image Base64 string for a job."""
    global current_previews
    return current_previews.get(job_id)


def clear_current_preview(job_id: str):
    """Removes the preview image Base64 string for a job."""
    global current_previews
    if job_id in current_previews:
        del current_previews[job_id]
        logging.info(f"Cleared preview for job {job_id}")


# --- Job Progress Update ---

def update_job_progress(job_id: str, progress: float, step: int, total: int, info: str):
    """Updates the progress fields of a job in the global queue and saves the file."""
    logging.info(f"Attempting to update progress for job {job_id}: progress={progress}, step={step}, total={total}, info='{info}'")
    global job_queue
    job_updated = False
    job_found_in_memory = False
    for job in job_queue:
        if job.job_id == job_id:
            # Update progress fields and timestamp
            needs_update = False
            if job.progress != progress:
                job.progress = progress
                needs_update = True
            if job.progress_step != step:
                job.progress_step = step
                needs_update = True
            if job.progress_total != total:
                job.progress_total = total
                needs_update = True
            if job.progress_info != info:
                job.progress_info = info
                needs_update = True

            if needs_update:
                job.updated_at = datetime.now(timezone.utc)
                job_updated = True
                job_found_in_memory = True
                logging.info(f"Job {job_id} found in memory. Updating progress and timestamp.")
            break

    if job_updated:
        logging.info(f"Progress updated for job {job_id} in memory. Attempting to save queue.")
        if save_queue():  # Save if updated in memory
            logging.info(f"Queue saved successfully after updating progress for job {job_id} in memory.")
        else:
            logging.error(f"Failed to save queue after updating progress for job {job_id} in memory.")

    # If not found or updated in memory, try loading from file, updating, and saving
    if not job_found_in_memory:
        logging.info(f"Job {job_id} not found in memory. Attempting to load from file.")
        current_queue = load_queue_from_file()
        job_found_in_file = False
        for job in current_queue:
            if job.job_id == job_id:
                logging.info(f"Job {job_id} found in file. Updating progress.")
                # Update progress fields and timestamp in file data
                needs_update = False
                if job.progress != progress:
                    job.progress = progress
                    needs_update = True
                if job.progress_step != step:
                    job.progress_step = step
                    needs_update = True
                if job.progress_total != total:
                    job.progress_total = total
                    needs_update = True
                if job.progress_info != info:
                    job.progress_info = info
                    needs_update = True

                if needs_update:
                    job.updated_at = datetime.now(timezone.utc)
                    job_found_in_file = True
                    logging.info(f"Job {job_id} found in file. Updating progress and timestamp.")
                break

        if job_found_in_file:
            logging.info(f"Progress updated for job {job_id} in file data. Attempting to save queue.")
            # Overwrite the file with the updated list
            try:
                jobs_to_save = [j.to_dict() for j in current_queue if j.to_dict() is not None]
                file_path = os.path.abspath(QUEUE_FILE)
                with open(file_path, 'w') as f:
                    json.dump(jobs_to_save, f, indent=2)
                job_updated = True  # Mark as updated since we saved the file
                logging.info(f"Queue saved successfully after updating progress for job {job_id} found in file.")
                # Update the global in-memory queue as well
                job_queue = current_queue  # Update global variable after successful save
                logging.info(f"Global job_queue updated with file content for job {job_id}.")
            except Exception as e:
                logging.error(f"Error saving queue after updating progress for job {job_id} found in file: {e}")
                traceback.print_exc()
                job_updated = False  # Ensure update status reflects save failure
        else:
            logging.warning(f"Job with ID {job_id} not found in memory or file for progress update.")

    if not job_updated:
        logging.warning(f"Progress update failed for job {job_id}.")
    return job_updated


def get_queue_status():
    """Returns a list of job statuses and basic info."""
    global job_queue
    # Load the latest queue to ensure we have the most recent list in memory
    # Note: This might overwrite changes made by other processes if not careful,
    # but it's necessary for get_queue_status to be accurate.
    # Consider locking mechanisms for multi-worker scenarios.
    load_queue_from_file()  # Reload global job_queue
    status_list = []
    for job in job_queue:  # Iterate over the reloaded global queue
        logging.info(f"Getting status for job {job.job_id}. Current progress: {job.progress}")  # Log progress here
        status_list.append({
            "job_id": job.job_id,
            "status": job.status,
            "prompt": job.prompt[:50] + "..." if len(job.prompt) > 50 else job.prompt,  # Truncate long prompts
            "video_length": job.video_length,
            # Include progress in status summary
            "progress": job.progress,
            "progress_info": job.progress_info,
            "job_type": job.job_type,  # Include job type
        })
    return status_list

# --- Functions below might be Gradio specific and potentially moved later or adapted ---


def update_queue_display():
    """
    This function seems tightly coupled with Gradio's display logic.
    For the API, we might need a different way to represent the queue,
    perhaps just returning the list of job dicts or a summary.
    Keeping it here for now, but marked for review.
    """
    global job_queue
    try:
        # Reload queue to ensure it's current
        load_queue_from_file()  # Reload global job_queue
        queue_data = []
        for job in job_queue:  # Iterate over the reloaded global queue
            # Create thumbnail if it doesn't exist and image path is valid
            if not job.thumbnail and job.image_path and os.path.exists(job.image_path):
                try:
                    # Load and resize image for thumbnail
                    img = Image.open(job.image_path)
                    width, height = img.size
                    new_height = 100  # Smaller thumbnail for potentially long queues
                    new_width = int((new_height / height) * width)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    thumb_path = os.path.join(temp_queue_images, f"thumb_{job.job_id}.png")
                    img.save(thumb_path)
                    job.thumbnail = thumb_path
                    save_queue()  # Save queue after updating thumbnail path
                except FileNotFoundError:
                    print(f"Image file not found for job {job.job_id} at path: {job.image_path}. Skipping thumbnail generation.")
                    job.thumbnail = ""  # Ensure thumbnail is empty if image is missing
                except Exception as e:
                    print(f"Error creating thumbnail for job {job.job_id}: {str(e)}")
                    job.thumbnail = ""  # Ensure thumbnail is empty on error
            elif not os.path.exists(job.image_path):
                print(f"Image file not found for job {job.job_id} at path: {job.image_path}. Skipping thumbnail display.")
                job.thumbnail = ""  # Ensure thumbnail is empty if image is missing

            # Add job data to display if thumbnail exists
            if job.thumbnail and os.path.exists(job.thumbnail):
                caption = f"ID: {job.job_id}\nStatus: {job.status}\nLen: {job.video_length}s\nPrompt: {job.prompt[:30]}..."
                queue_data.append((job.thumbnail, caption))
            # Optionally, add placeholder if no thumbnail
            # else:
            #     caption = f"ID: {job.job_id}\nStatus: {job.status}\nLen: {job.video_length}s\nPrompt: {job.prompt[:30]}...\n(No Thumbnail)"
            #     # You might need a placeholder image path or handle this differently in Gradio
            #     placeholder_thumb = "path/to/placeholder.png" # Define a placeholder image
            #     if os.path.exists(placeholder_thumb):
        # End of the try block from line 523
        return queue_data  # Return data collected in the try block
    except Exception as e:  # Add except block for the try at line 523
        print(f"Error updating queue display: {str(e)}")
        # traceback.print_exc()  # Consider removing or using logging for production
        return []  # Return empty list on error


# --- Cleanup Function ---


def cleanup_jobs_by_max_count(max_completed_jobs: int = settings.MAX_COMPLETED_JOBS):
    """
    Removes old completed, cancelled, or failed jobs if their total count exceeds the limit.
    Jobs are removed based on their 'updated_at' timestamp (oldest first).
    Also removes associated temporary files (input image, thumbnail).
    """
    global job_queue
    logging.info(f"Starting job cleanup. Max completed/cancelled/failed jobs to keep: {max_completed_jobs}")

    try:
        current_queue = load_queue_from_file()
        # terminal_statuses = {"completed", "cancelled", "failed"} # Unused variable

        # Separate jobs by status
        active_jobs = []  # pending, processing
        terminal_jobs = []  # completed, cancelled, failed

        for job in current_queue:
            # Handle potential variations in 'failed' status (e.g., "failed - Reason")
            is_terminal = job.status == "completed" or job.status == "cancelled" or job.status.startswith("failed")
            if is_terminal:
                terminal_jobs.append(job)
            else:
                active_jobs.append(job)

        num_terminal_jobs = len(terminal_jobs)
        logging.info(f"Found {num_terminal_jobs} terminal jobs (completed/cancelled/failed).")

        if num_terminal_jobs <= max_completed_jobs:
            logging.info("Number of terminal jobs does not exceed the limit. No cleanup needed.")
            return 0  # No jobs removed

        # Sort terminal jobs by updated_at timestamp (oldest first)
        # Handle potential None values for updated_at just in case
        terminal_jobs.sort(key=lambda j: j.updated_at or datetime.min.replace(tzinfo=timezone.utc))

        # Determine how many jobs to remove
        num_to_remove = num_terminal_jobs - max_completed_jobs
        jobs_to_remove = terminal_jobs[:num_to_remove]
        jobs_to_keep = terminal_jobs[num_to_remove:]

        logging.info(f"Exceeded limit by {num_to_remove} jobs. Preparing to remove oldest ones.")

        removed_count = 0
        files_to_delete = []

        # Identify files associated with jobs to be removed
        for job in jobs_to_remove:
            logging.info(f"Marking job {job.job_id} (status: {job.status}, updated: {job.updated_at}) for removal.")
            if job.image_path:
                files_to_delete.append(job.image_path)
            if job.thumbnail:
                files_to_delete.append(job.thumbnail)
            # Also clear any lingering preview data for the removed job
            clear_current_preview(job.job_id)
            removed_count += 1

        # Combine kept terminal jobs and active jobs for the new queue
        new_queue_jobs = active_jobs + jobs_to_keep

        # Save the new queue to the file
        try:
            jobs_to_save = [j.to_dict() for j in new_queue_jobs if j.to_dict() is not None]
            file_path = os.path.abspath(QUEUE_FILE)
            with open(file_path, 'w') as f:
                json.dump(jobs_to_save, f, indent=2)
            logging.info(f"Successfully saved updated queue file with {len(new_queue_jobs)} jobs.")
            # Update the global in-memory queue as well
            job_queue = new_queue_jobs
        except Exception as e:
            logging.error(f"Error saving queue file during cleanup: {e}")
            traceback.print_exc()
            # If saving fails, we should not proceed with file deletion
            return 0

        # Delete associated files *after* successfully saving the queue
        deleted_files_count = 0
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted temporary file: {file_path}")
                    deleted_files_count += 1
                else:
                    logging.warning(f"Temporary file not found for deletion: {file_path}")
            except Exception as e:
                logging.error(f"Error deleting temporary file {file_path}: {e}")

        logging.info(f"Job cleanup finished. Removed {removed_count} job entries and attempted to delete {deleted_files_count} associated files.")
        return removed_count

    except Exception as e:
        logging.error(f"An unexpected error occurred during job cleanup: {e}")
        traceback.print_exc()
        return 0
# Removed erroneous code from previous diff attempts