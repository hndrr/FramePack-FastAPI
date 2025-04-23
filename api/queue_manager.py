import os
import json
import traceback
import uuid
import numpy as np
from dataclasses import dataclass
from PIL import Image

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

    def to_dict(self):
        try:
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
                'mp4_crf': self.mp4_crf
            }
        except Exception as e:
            print(f"Error converting job to dict: {str(e)}")
            return None

    @classmethod
    def from_dict(cls, data):
        try:
            return cls(
                prompt=data.get('prompt', ''),
                image_path=data.get('image_path', ''),
                video_length=data.get('video_length', 5.0),
                job_id=data.get('job_id', uuid.uuid4().hex[:8]),  # Provide default if missing
                seed=data.get('seed', -1),
                use_teacache=data.get('use_teacache', False),
                gpu_memory_preservation=data.get('gpu_memory_preservation', 0.0),
                steps=data.get('steps', 20),
                cfg=data.get('cfg', 7.0),
                gs=data.get('gs', 1.0),
                rs=data.get('rs', 1.0),
                status=data.get('status', 'pending'),
                thumbnail=data.get('thumbnail', ''),
                mp4_crf=data.get('mp4_crf', 16.0)
            )
        except Exception as e:
            print(f"Error creating job from dict: {str(e)}")
            traceback.print_exc()
            return None


# Initialize job queue as a list
job_queue = []


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


def load_queue():
    global job_queue
    try:
        if os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'r') as f:
                jobs_data = json.load(f)
            # Clear existing queue and load jobs from file
            current_queue = []
            for job_data in jobs_data:
                job = QueuedJob.from_dict(job_data)
                if job is not None:
                    current_queue.append(job)
            job_queue = current_queue  # Assign the loaded queue
            return job_queue
        return []
    except Exception as e:
        print(f"Error loading queue: {str(e)}")
        traceback.print_exc()
        return []


# Load existing queue on startup
job_queue = load_queue()


def save_image_to_temp(image: np.ndarray, job_id: str) -> str:
    """Save image to temp directory and return the path"""
    try:
        # Convert numpy array to PIL Image
        # Remove single-dimensional entries from the shape of an array
        squeezed_image = np.squeeze(image)
        pil_image = Image.fromarray(squeezed_image)
        # Create unique filename using hex ID
        filename = f"queue_image_{job_id}.png"
        filepath = os.path.join(temp_queue_images, filename)
        # Save image
        pil_image.save(filepath)
        return filepath
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return ""


def add_to_queue(prompt, image, video_length, seed, use_teacache, gpu_memory_preservation, steps, cfg, gs, rs, status="pending", mp4_crf=16):
    global job_queue
    try:
        # Generate a unique hex ID for the job
        job_id = uuid.uuid4().hex[:8]
        # Save image to temp directory and get path
        image_array = np.array(image)
        image_path = save_image_to_temp(image_array, job_id)
        if not image_path:
            print("Failed to save image to temp, cannot add job.")
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
            mp4_crf=mp4_crf
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


def get_next_job():
    global job_queue
    try:
        if job_queue:
            job = job_queue.pop(0)  # Remove and return first job
            save_queue()  # Save after removing job
            return job
        return None
    except Exception as e:
        print(f"Error getting next job: {str(e)}")
        traceback.print_exc()
        return None


def get_job_by_id(job_id: str) -> QueuedJob | None:
    global job_queue
    for job in job_queue:
        if job.job_id == job_id:
            return job
    # If not in memory queue, try loading from file again (in case another process updated it)
    load_queue()
    for job in job_queue:
        if job.job_id == job_id:
            return job
    return None


def update_job_status(job_id: str, status: str, thumbnail: str = None):
    global job_queue
    job_updated = True  # Assume update will succeed initially
    for job in job_queue:
        if job.job_id == job_id:
            job.status = status
            if thumbnail:
                job.thumbnail = thumbnail
            job_updated = True
            break
    if job_updated:
        save_queue()
    else:
        print(f"Job with ID {job_id} not found in queue for status update.")
    return job_updated


def get_queue_status():
    """Returns a list of job statuses and basic info."""
    global job_queue
    # Ensure the queue is up-to-date
    load_queue()
    status_list = []
    for job in job_queue:
        status_list.append({
            "job_id": job.job_id,
            "status": job.status,
            "prompt": job.prompt[:50] + "..." if len(job.prompt) > 50 else job.prompt,  # Truncate long prompts
            "video_length": job.video_length
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
        load_queue()
        queue_data = []
        for job in job_queue:
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
            #         queue_data.append((placeholder_thumb, caption))

        return queue_data
    except Exception as e:
        print(f"Error updating queue display: {str(e)}")
        traceback.print_exc()
        return []