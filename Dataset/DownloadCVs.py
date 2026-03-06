import kagglehub
from pathlib import Path

dataset_path = Path("Input")
dataset_path.mkdir(parents=True, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download("snehaanbhawal/resume-dataset", output_dir=str(dataset_path))

print("Path to dataset files:", path)