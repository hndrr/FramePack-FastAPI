import os
from dotenv import load_dotenv

# Load environment variables from .env file located in the project root
# Assumes settings.py is in 'api' subdirectory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# API Server Settings
API_HOST = "0.0.0.0"  # Listen on all available network interfaces
API_PORT = int(os.environ.get("API_PORT", 8080))  # Load from env var, default to 8080

# Model Settings (Consider making these paths absolute or relative to a known root)
# LORA_PATH = None  # Example: Set path if LoRA is always used, or load from env/config
LORA_PATH = os.environ.get("LORA_PATH", None)  # Load LoRA path from environment variable if set

# Directory Settings (Relative to project root, one level up from this file's directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
TEMP_QUEUE_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'temp_queue_images')
QUEUE_FILE_PATH = os.path.join(PROJECT_ROOT, 'job_queue.json')
HF_HOME_DIR = os.path.join(PROJECT_ROOT, 'hf_download')
LORA_DIR = os.environ.get("LORA_DIR", os.path.join(PROJECT_ROOT, 'loras'))  # 追加: LoRAディレクトリ


# Ensure directories exist (These should ideally be created outside the API module if they don't exist)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_QUEUE_IMAGES_DIR, exist_ok=True)
os.makedirs(HF_HOME_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)  # 追加: LoRAディレクトリも作成

# Set Hugging Face home directory environment variable
os.environ['HF_HOME'] = HF_HOME_DIR

# Worker Settings
# How often the background worker checks the queue (in seconds)
WORKER_CHECK_INTERVAL = 5

# VRAM Settings (can be overridden by detection in models.py)
# HIGH_VRAM_THRESHOLD_GB = 60  # Example threshold

print("Settings loaded:")  # Removed f-string as there are no placeholders
print(f"  API Host: {API_HOST}")
print(f"  API Port: {API_PORT}")
print(f"  LoRA Path: {LORA_PATH}")
print(f"  Outputs Dir: {OUTPUTS_DIR}")
print(f"  Temp Queue Images Dir: {TEMP_QUEUE_IMAGES_DIR}")
print(f"  Queue File Path: {QUEUE_FILE_PATH}")
print(f"  HF Home Dir: {HF_HOME_DIR}")
print(f"  LoRA Dir: {LORA_DIR}")  # 追加
print(f"  Worker Check Interval: {WORKER_CHECK_INTERVAL}")
