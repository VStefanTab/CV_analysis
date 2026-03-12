import kagglehub
from pathlib import Path
import subprocess

def downloadFromKaggle():
    datasets = [
        "snehaanbhawal/resume-dataset",
        "saugataroyarghya/resume-dataset"
    ]

    dataset_paths = []

    for i, _ in enumerate(datasets):
        dataset_paths.append(Path("Input")/ "Kaggle" / f"dataset{i}")

    for path in dataset_paths:
        path.mkdir(parents=True, exist_ok=True)

    # Download latest version
    for el in zip(datasets, dataset_paths):
        print(f"Downloading: {el[0]} in {el[1]}")
        path =  kagglehub.dataset_download(el[0], output_dir=str(el[1]))

def downloadFromGit():
    dataset_paths = []

if __name__ == "__main__":
    downloadFromKaggle()
    downloadFromGit()