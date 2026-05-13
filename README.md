# CV-Job Description matching

## How to install
First create a python environment
```sh
python -m venv .venv
# windows
.venv\Scripts\activate.bat
# linux
.venv/bin/activate
```

Install the dependencies
```sh
pip install -r requirements.txt
```

Get Kaggle credentials to download the dataset
- Access Kaggle
- Go to API Tokens
- Create legacy api key
- Download a .json file with username and secret key
- On colab
- Go to secrets Tab -> Add KAGGLE_USERNAME and KAGGLE_API_TOKEN from .json file

Run the colab (if you do not have a `best_model.pt` in `assets/`)
- Go into the GDrive Collab and run each container
- At the end it will download a .zip file
- Unzip the content into `assets/`

Run the streamlit application
```sh
cd src
streamlit run app.py
```