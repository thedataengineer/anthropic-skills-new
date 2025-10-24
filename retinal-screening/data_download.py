import kaggle
import os

# Download Kaggle Diabetic Retinopathy dataset
# Note: You need to have kaggle API key set up
# kaggle competitions download -c diabetic-retinopathy-detection

def download_dataset():
    if not os.path.exists('data'):
        os.makedirs('data')

    # Download the dataset
    # This requires kaggle API
    # kaggle.api.competition_download_files('diabetic-retinopathy-detection', path='data', unzip=True)

    print("Download the dataset from https://www.kaggle.com/c/diabetic-retinopathy-detection/data")
    print("Unzip and organize into data/train/ and data/val/ folders")

if __name__ == "__main__":
    download_dataset()
