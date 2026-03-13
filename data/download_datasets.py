from pathlib import Path
import kagglehub

def download_from_kaggle(datasets, dataset_type):

    dataset_paths = []

    for i, _ in enumerate(datasets):
        dataset_paths.append(Path("input")/ dataset_type / f"dataset{i}")

    for path in dataset_paths:
        path.mkdir(parents=True, exist_ok=True)

    # Download latest version
    for el in zip(datasets, dataset_paths):
        print(f"Downloading: {el[0]} in {el[1]}")
        path =  kagglehub.dataset_download(el[0], output_dir=str(el[1]))

def download_from_git():
    pass

if __name__ == "__main__":
    datasets_cvs = [
        "snehaanbhawal/resume-dataset",
        "saugataroyarghya/resume-dataset",

    ]
    datasets_jobs = [
        "arshkon/linkedin-job-postings"
    ]
    download_from_kaggle(datasets_cvs, "cvs")
    download_from_kaggle(datasets_jobs, "jobs")
    # download_from_git()