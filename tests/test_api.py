import pytest
import torch
from fastapi.testclient import TestClient
import io
import os
import sys  # Import sys to modify sys.modules
import base64
# import asyncio # Not directly used when only AsyncMock is needed
from unittest.mock import mock_open
from PIL import Image
from api import queue_manager, settings  # settings をインポート

# Assuming your FastAPI app instance is named 'app' in 'api/api.py'
from api.api import app


# Create a TestClient instance
client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_heavy_operations(mocker):
    """
    Fixture to automatically mock heavy operations for all tests.
    - Mocks model loading to return a dummy dict.
    - Mocks the background worker task to prevent it from starting.
    - Mocks GPU/CUDA related functions to avoid errors in environments without NVIDIA drivers.
    """
    # --- Mock problematic imports causing fatal errors ---
    # Mock the entire sageattention module and its submodules early in sys.modules
    # to prevent DLL loading errors during import or patching attempts later.
    sys.modules['sageattention'] = mocker.MagicMock()
    sys.modules['sageattention.core'] = mocker.MagicMock()
    sys.modules['sageattention.quant'] = mocker.MagicMock()
    # Mock triton as well, as it might be another source of import/DLL issues in test env
    sys.modules['triton'] = mocker.MagicMock()
    sys.modules['triton.language'] = mocker.MagicMock()    # Mock submodules if accessed directly

    # Mock model loading
    mocker.patch.dict("api.api.loaded_models", {"model": "mocked"}, clear=True)
    # Mock background worker
    mocker.patch("api.api.background_worker_task")
    # Mock model unloading during shutdown
    mocker.patch("api.models.unload_models")

    # --- Mock GPU/CUDA related parts ---
    # Mock the gpu device object itself to avoid cuda initialization error at import time
    # Need to mock where it's defined and potentially where it's used if imported directly
    mocker.patch("diffusers_helper.memory.gpu", torch.device('cpu'))  # Use cpu device
    mocker.patch("api.models.gpu", torch.device('cpu'))  # Mock in models module too if imported there

    # Mock functions that interact with CUDA memory or device properties
    mocker.patch("diffusers_helper.memory.get_cuda_free_memory_gb", return_value=16.0)  # Return dummy high value
    mocker.patch("diffusers_helper.memory.torch.cuda.is_available", return_value=False)
    mocker.patch("diffusers_helper.memory.torch.cuda.current_device", return_value=0)  # Return dummy device index
    mocker.patch("diffusers_helper.memory.torch.cuda.empty_cache")  # Mock to do nothing
    mocker.patch("diffusers_helper.memory.torch.cuda.mem_get_info", return_value=(16 * 1024 ** 3, 16 * 1024 ** 3))
    mocker.patch("diffusers_helper.memory.torch.cuda.memory_stats", return_value={'active_bytes.all.current': 0, 'reserved_bytes.all.current': 0})

    # Mock functions that move models (they might still be called by logic we don't bypass)
    mocker.patch("diffusers_helper.memory.move_model_to_device_with_memory_preservation")
    mocker.patch("diffusers_helper.memory.offload_model_from_device_for_memory_preservation")
    mocker.patch("diffusers_helper.memory.load_model_as_complete")
    mocker.patch("diffusers_helper.memory.unload_complete_models")

    # Mock DynamicSwapInstaller methods if they interact with CUDA implicitly or cause issues
    mocker.patch("diffusers_helper.memory.DynamicSwapInstaller.install_model")
    mocker.patch("diffusers_helper.memory.DynamicSwapInstaller.uninstall_model")

    # No longer need specific extension mocks as the parent modules are mocked


# Helper function to create a dummy image for uploads
def create_dummy_image(width=100, height=50, format="PNG"):
    """Creates an in-memory dummy image file."""
    img = Image.new('RGB', (width, height), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr


@pytest.mark.parametrize(
    "lora_path_param, lora_scale_param, expected_lora_path_in_queue, expected_lora_scale_in_queue",
    [
        ("test_lora.safetensors", 0.8, "test_lora.safetensors", 0.8),
        ("another_lora.pt", None, "another_lora.pt", 1.0),
        (None, None, None, 1.0),
    ]
)
def test_generate_job_success(mocker, lora_path_param, lora_scale_param, expected_lora_path_in_queue, expected_lora_scale_in_queue):
    """
    Test successful job submission via the /generate endpoint with different LoRA parameters.
    Mocks the actual worker and queue manager add function.
    """
    # Mock the queue manager's add_to_queue function to return a predictable job_id
    # and prevent actual file saving/queue modification during test.
    mock_job_id = f"testjob_{lora_path_param}_{lora_scale_param}"
    mock_add_to_queue = mocker.patch("api.queue_manager.add_to_queue", return_value=mock_job_id)

    # Mock the worker function so it doesn't actually run the heavy process
    mock_worker = mocker.patch("api.worker.worker")

    # Prepare dummy image data
    dummy_image_file = create_dummy_image()
    files = {'image': ('dummy.png', dummy_image_file, 'image/png')}

    # Prepare form data
    data = {
        "prompt": "A test prompt",
        "video_length": 1.0,
        "seed": 12345,
        "steps": 5,
        "use_teacache": True,
        "gpu_memory_preservation": 6.0,
        "cfg": 7.0,
        "gs": 1.0,
        "rs": 1.0,
        "mp4_crf": 18.0,
    }
    # Add LoRA params if they are not None
    if lora_path_param is not None:
        data["lora_path"] = lora_path_param
    if lora_scale_param is not None:
        data["lora_scale"] = lora_scale_param

    # Send the POST request
    response = client.post("/generate", data=data, files=files)

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["job_id"] == mock_job_id
    assert response_json["message"] == "Video generation job added to queue."

    # Verify that add_to_queue was called with the correct arguments
    mock_add_to_queue.assert_called_once_with(
        prompt=data["prompt"],
        image=mocker.ANY,
        original_exif=None,
        video_length=data["video_length"],
        seed=data["seed"],
        use_teacache=data["use_teacache"],
        gpu_memory_preservation=data["gpu_memory_preservation"],
        steps=data["steps"],
        cfg=data["cfg"],
        gs=data["gs"],
        rs=data["rs"],
        mp4_crf=data["mp4_crf"],
        lora_scale=expected_lora_scale_in_queue,
        lora_path=expected_lora_path_in_queue,
        status="pending"
    )

    # Verify that the worker was NOT called directly by the endpoint
    mock_worker.assert_not_called()


def test_get_status_pending(mocker):
    """Test getting status for a pending job."""
    mock_job_id = "pendingjob456"
    # Create a mock job object with pending status
    mock_job_data = queue_manager.QueuedJob(
        prompt="pending prompt",
        image_path="/fake/path.png",
        video_length=2.0,
        job_id=mock_job_id,
        seed=67890,
        use_teacache=False,
        gpu_memory_preservation=0.0,
        steps=10,
        cfg=7.0, gs=1.0, rs=1.0,
        status="pending",
        mp4_crf=16.0,
    )

    # Mock get_job_by_id to return our mock job data
    mocker.patch("api.queue_manager.get_job_by_id", return_value=mock_job_data)
    # Mock os.path.exists for the output file check to return False
    mocker.patch("os.path.exists", return_value=False)
    # Mock the global variable for currently processing job ID
    mocker.patch("api.api.currently_processing_job_id", None)

    # Send the GET request
    response = client.get(f"/status/{mock_job_id}")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["job_id"] == mock_job_id
    assert response_json["status"] == "pending"
    assert response_json["progress"] == 0.0
    assert response_json["progress_info"] == ""

    # Verify mocks were called as expected
    queue_manager.get_job_by_id.assert_called_once_with(mock_job_id)


def test_get_status_processing(mocker):
    """Test getting status for a processing job."""
    mock_job_id = "processingjob789"
    # Create a mock job object with processing status and some progress
    mock_job_data = queue_manager.QueuedJob(
        prompt="processing prompt",
        image_path="/fake/process.png",
        video_length=3.0,
        job_id=mock_job_id,
        seed=11223,
        use_teacache=True,
        gpu_memory_preservation=6.0,
        steps=20,
        cfg=7.0, gs=1.0, rs=1.0,
        status="processing",
        mp4_crf=16.0,
        progress=55.5,
        progress_step=11,
        progress_total=20,
        progress_info="Sampling section 1/2 - Step 11/20"
    )

    # Mock get_job_by_id to return our mock job data
    mocker.patch("api.queue_manager.get_job_by_id", return_value=mock_job_data)
    # Mock os.path.exists for the output file check to return False
    mocker.patch("os.path.exists", return_value=False)
    # Mock the global variable to indicate this job IS currently processing
    mocker.patch("api.api.currently_processing_job_id", mock_job_id)

    # Send the GET request
    response = client.get(f"/status/{mock_job_id}")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["job_id"] == mock_job_id
    assert response_json["status"] == "processing"
    assert response_json["progress"] == 55.5
    assert response_json["progress_step"] == 11
    assert response_json["progress_total"] == 20
    assert response_json["progress_info"] == "Sampling section 1/2 - Step 11/20"

    # Verify mocks
    queue_manager.get_job_by_id.assert_called_once_with(mock_job_id)


def test_get_status_completed(mocker):
    """Test getting status for a completed job (output file exists)."""
    mock_job_id = "completedjobABC"
    # Mock get_job_by_id to return None (or a completed job, either works for this path)
    mocker.patch("api.queue_manager.get_job_by_id", return_value=None)
    # Mock os.path.exists for the output file check to return True
    mocker.patch("os.path.exists", return_value=True)
    # Mock the global variable for currently processing job ID
    mocker.patch("api.api.currently_processing_job_id", None)

    # Send the GET request
    response = client.get(f"/status/{mock_job_id}")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["job_id"] == mock_job_id
    assert response_json["status"] == "completed"
    assert response_json["progress"] == 100.0
    assert response_json["progress_info"] == "Completed"

    # Verify mocks
    queue_manager.get_job_by_id.assert_called_once_with(mock_job_id)


def test_get_status_not_found(mocker):
    """Test getting status for a non-existent job."""
    mock_job_id = "notfoundjobXYZ"
    # Mock get_job_by_id to return None
    mocker.patch("api.queue_manager.get_job_by_id", return_value=None)
    # Mock os.path.exists for the output file check to return False
    mocker.patch("os.path.exists", return_value=False)
    # Mock the global variable for currently processing job ID
    mocker.patch("api.api.currently_processing_job_id", None)

    # Send the GET request
    response = client.get(f"/status/{mock_job_id}")

    # Assertions
    assert response.status_code == 404
    response_json = response.json()
    assert response_json["detail"] == "Job not found"

    # Verify mocks
    queue_manager.get_job_by_id.assert_called_once_with(mock_job_id)


def test_get_result_completed(mocker):
    """Test getting the result for a completed job."""
    mock_job_id = "resultjobDEF"
    # Mock job data (status completed)
    mock_job_data = queue_manager.QueuedJob(
        prompt="completed prompt", image_path="/fake/completed.png", video_length=1.0,
        job_id=mock_job_id, seed=1, use_teacache=True, gpu_memory_preservation=0,
        steps=1, cfg=1, gs=1, rs=1, status="completed", mp4_crf=16,
        thumbnail="/fake/thumb_resultjobDEF.jpg"  # サムネイルパスを追加
    )
    # Mock get_job_by_id to return the completed job
    mocker.patch("api.queue_manager.get_job_by_id", return_value=mock_job_data)

    # Mock os.path.exists to return True for both video and thumbnail
    def mock_exists(path):
        if path == os.path.join(settings.OUTPUTS_DIR, f"{mock_job_id}.mp4"):
            return True
        if path == mock_job_data.thumbnail:
            return True
        return False
    mocker.patch("os.path.exists", side_effect=mock_exists)

    # Mock built-in open for reading the thumbnail file
    dummy_thumbnail_data = b"dummy_jpeg_data"
    mock_file = mock_open(read_data=dummy_thumbnail_data)
    # Patch open in the context where it's used (api.api)
    mocker.patch("api.api.open", mock_file)

    # Mock mimetypes.guess_type and capture the mock object
    mock_guess_type = mocker.patch("api.api.mimetypes.guess_type", return_value=("image/jpeg", None))

    # Send the GET request
    # TestClient automatically uses a base URL like http://testserver
    response = client.get(f"/result/{mock_job_id}")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert "video_url" in response_json
    assert "thumbnail_base64" in response_json

    # Check video URL (assuming default TestClient base URL)
    expected_video_url = f"http://testserver/download/video/{mock_job_id}"
    assert response_json["video_url"] == expected_video_url

    # Check thumbnail Base64 data
    expected_base64_data = base64.b64encode(dummy_thumbnail_data).decode("utf-8")
    expected_thumbnail_base64 = f"data:image/jpeg;base64,{expected_base64_data}"
    assert response_json["thumbnail_base64"] == expected_thumbnail_base64

    # Verify mocks related to thumbnail reading
    mock_file.assert_called_once_with(mock_job_data.thumbnail, "rb")
    mock_guess_type.assert_called_once_with(mock_job_data.thumbnail)  # Use the captured mock object


def test_get_result_not_completed(mocker):
    """Test getting the result for a job that is not completed."""
    mock_job_id = "notyetjobGHI"
    # Mock job data (status pending)
    mock_job_data = queue_manager.QueuedJob(
        prompt="pending prompt", image_path="/fake/pending.png", video_length=1.0,
        job_id=mock_job_id, seed=1, use_teacache=True, gpu_memory_preservation=0,
        steps=1, cfg=1, gs=1, rs=1, status="pending", mp4_crf=16
    )
    # Mock get_job_by_id to return the pending job
    mocker.patch("api.queue_manager.get_job_by_id", return_value=mock_job_data)
    # Mock os.path.exists for the output file check to return False
    mocker.patch("os.path.exists", return_value=False)

    # Send the GET request
    response = client.get(f"/result/{mock_job_id}")

    # Assertions
    assert response.status_code == 404
    response_json = response.json()
    assert response_json["detail"] == f"Job '{mock_job_id}' is not completed yet (status: pending)."

    # Verify mocks
    queue_manager.get_job_by_id.assert_called_once_with(mock_job_id)


def test_get_result_not_found(mocker):
    """Test getting the result for a non-existent job."""
    mock_job_id = "neverexistedJKL"
    # Mock get_job_by_id to return None
    mocker.patch("api.queue_manager.get_job_by_id", return_value=None)
    # Mock os.path.exists for the output file check to return False
    mocker.patch("os.path.exists", return_value=False)

    # Send the GET request
    response = client.get(f"/result/{mock_job_id}")

    # Assertions
    assert response.status_code == 404
    response_json = response.json()
    # The exact error message might vary slightly, adjust if needed
    assert f"Job '{mock_job_id}' not found" in response_json["detail"]

    # Verify mocks
    queue_manager.get_job_by_id.assert_called_once_with(mock_job_id)


def test_cancel_job_success(mocker):
    """Test successfully requesting cancellation for a pending job."""
    mock_job_id = "cancellingjobMNO"
    # Mock job data (status pending)
    mock_job_data = queue_manager.QueuedJob(
        prompt="to be cancelled", image_path="/fake/cancel.png", video_length=1.0,
        job_id=mock_job_id, seed=1, use_teacache=True, gpu_memory_preservation=0,
        steps=1, cfg=1, gs=1, rs=1, status="pending", mp4_crf=16
    )
    # Mock get_job_by_id to return the pending job initially
    mocker.patch("api.queue_manager.get_job_by_id", return_value=mock_job_data)
    # Mock os.path.exists for the output file check to return False
    mocker.patch("os.path.exists", return_value=False)
    # Mock update_job_status to check if it's called correctly
    mock_update_status = mocker.patch("api.queue_manager.update_job_status", return_value=True)

    # Send the POST request
    response = client.post(f"/cancel/{mock_job_id}")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["message"] == f"Cancellation requested for job {mock_job_id}."

    # Verify that update_job_status was called with "cancelled"
    mock_update_status.assert_called_once_with(mock_job_id, "cancelled")


def test_cancel_job_already_completed(mocker):
    """Test requesting cancellation for an already completed job."""
    mock_job_id = "alreadycompletedPQR"
    # Mock job data (status completed)
    mock_job_data = queue_manager.QueuedJob(
        prompt="already done", image_path="/fake/done.png", video_length=1.0,
        job_id=mock_job_id, seed=1, use_teacache=True, gpu_memory_preservation=0,
        steps=1, cfg=1, gs=1, rs=1, status="completed", mp4_crf=16
    )
    # Mock get_job_by_id to return the completed job
    mocker.patch("api.queue_manager.get_job_by_id", return_value=mock_job_data)
    # Mock os.path.exists for the output file check (might be called, return True just in case)
    mocker.patch("os.path.exists", return_value=True)
    # Mock update_job_status to ensure it's NOT called
    mock_update_status = mocker.patch("api.queue_manager.update_job_status")

    # Send the POST request
    response = client.post(f"/cancel/{mock_job_id}")

    # Assertions
    assert response.status_code == 200  # Endpoint itself succeeds
    response_json = response.json()
    assert response_json["message"] == "Job is already completed."

    # Verify that update_job_status was NOT called
    mock_update_status.assert_not_called()


def test_cancel_job_not_found(mocker):
    """Test requesting cancellation for a non-existent job."""
    mock_job_id = "ghostjobSTU"
    # Mock get_job_by_id to return None
    mocker.patch("api.queue_manager.get_job_by_id", return_value=None)
    # Mock os.path.exists for the output file check to return False
    mocker.patch("os.path.exists", return_value=False)
    # Mock update_job_status to ensure it's NOT called
    mock_update_status = mocker.patch("api.queue_manager.update_job_status")

    # Send the POST request
    response = client.post(f"/cancel/{mock_job_id}")

    # Assertions
    assert response.status_code == 404
    response_json = response.json()
    assert response_json["detail"] == "Job not found"

    # Verify that update_job_status was NOT called
    mock_update_status.assert_not_called()


def test_get_worker_status_processing(mocker):
    """Test getting worker status when it's processing a job."""
    mock_job_id = "workerjobVWX"
    # Mock the global variables in api.api
    mocker.patch("api.api.worker_running", True)
    mocker.patch("api.api.currently_processing_job_id", mock_job_id)

    # Send the GET request
    response = client.get("/worker/status")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["is_running"] is True
    assert response_json["processing_job_id"] == mock_job_id


def test_get_worker_status_idle(mocker):
    """Test getting worker status when it's idle."""
    # Mock the global variables in api.api
    mocker.patch("api.api.worker_running", True)  # Still running, but no job
    mocker.patch("api.api.currently_processing_job_id", None)

    # Send the GET request
    response = client.get("/worker/status")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["is_running"] is True
    assert response_json["processing_job_id"] is None


def test_get_worker_status_not_running(mocker):
    """Test getting worker status when it's not running."""
    # Mock the global variables in api.api
    mocker.patch("api.api.worker_running", False)
    mocker.patch("api.api.currently_processing_job_id", None)  # Should be None if not running

    # Send the GET request
    response = client.get("/worker/status")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["is_running"] is False
    assert response_json["processing_job_id"] is None


# --- Tests for /loras endpoint ---

def test_list_loras_success(mocker):
    """Test the /loras endpoint successfully lists files."""
    # Mock os.listdir to return dummy files
    mock_lora_files = ["lora1.safetensors", "lora2.pt", "other_file.txt", "lora3.bin"]
    mocker.patch("os.listdir", return_value=mock_lora_files)
    # Mock os.path.isfile to return True for relevant files

    def mock_isfile(path):
        filename = os.path.basename(path)
        # Check if the filename is in our mock list and has an extension
        return filename in mock_lora_files and "." in filename
    mocker.patch("os.path.isfile", side_effect=mock_isfile)
    # Mock os.path.isdir to return True for the LORA_DIR
    mocker.patch("os.path.isdir", return_value=True)
    # Mock settings.LORA_DIR if needed, or assume it's set correctly
    # mocker.patch("api.settings.LORA_DIR", "/fake/lora/dir")

    response = client.get("/loras")

    assert response.status_code == 200
    response_json = response.json()
    # Should only contain files with allowed extensions, sorted
    expected_loras = ["lora1.safetensors", "lora2.pt", "lora3.bin"]
    assert response_json["loras"] == expected_loras
    # Verify listdir was called with the correct directory from settings
    os.listdir.assert_called_once_with(settings.LORA_DIR)


def test_list_loras_dir_not_found(mocker):
    """Test the /loras endpoint when the LORA_DIR doesn't exist."""
    # Mock os.path.isdir to return False
    mocker.patch("os.path.isdir", return_value=False)
    mock_listdir = mocker.patch("os.listdir")  # Ensure listdir is not called

    response = client.get("/loras")

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["loras"] == []  # Should return empty list
    mock_listdir.assert_not_called()  # listdir shouldn't be called if isdir is false


def test_list_loras_exception(mocker):
    """Test the /loras endpoint handles exceptions during listing."""
    # Mock os.listdir to raise an exception
    mocker.patch("os.listdir", side_effect=PermissionError("Test permission error"))
    mocker.patch("os.path.isdir", return_value=True)  # Assume directory exists

    response = client.get("/loras")

    # The endpoint currently catches exceptions and returns empty list
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["loras"] == []


# --- Test for /queue endpoint ---

def test_get_queue_info(mocker):
    """Test the /queue endpoint successfully returns queue status."""
    # Define dummy queue data that get_queue_status might return
    mock_queue_data = [
        {"job_id": "job1", "status": "pending", "prompt": "prompt1...", "video_length": 5.0, "progress": 0.0, "progress_info": ""},
        {"job_id": "job2", "status": "processing", "prompt": "prompt2...", "video_length": 2.0, "progress": 50.0, "progress_info": "Sampling..."},
        {"job_id": "job3", "status": "completed", "prompt": "prompt3...", "video_length": 1.0, "progress": 100.0, "progress_info": "Completed"},
    ]
    # Mock the function called by the endpoint
    mocker.patch("api.queue_manager.get_queue_status", return_value=mock_queue_data)

    # Call the endpoint
    response = client.get("/queue")

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["queue"] == mock_queue_data

    # Verify the mock was called
    queue_manager.get_queue_status.assert_called_once()
